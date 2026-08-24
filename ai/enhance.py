import os
import json
import sys
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
from queue import Queue
from threading import Lock
# INSERT_YOUR_CODE
import requests

import dotenv
import argparse
from tqdm import tqdm

import langchain_core.exceptions
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from structure import Structure

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from daily_arxiv.daily_arxiv.journal_rankings import (  # noqa: E402
    annotate_paper_journal_rank,
    journal_tier_distribution,
    select_journal_quality_and_exploration,
    selection_slots,
)

MAX_AI_PAPERS_PER_RUN = 20

if os.path.exists('.env'):
    dotenv.load_dotenv()
template = open("template.txt", "r").read()
system = open("system.txt", "r").read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def parse_filter_keywords(raw_keywords: Optional[str]) -> List[str]:
    """Parse comma-separated keywords and remove empty/case-insensitive duplicates."""
    if not raw_keywords:
        return []

    keywords = []
    seen = set()
    for raw_keyword in raw_keywords.split(","):
        keyword = raw_keyword.strip()
        normalized_keyword = keyword.casefold()
        if keyword and normalized_keyword not in seen:
            keywords.append(keyword)
            seen.add(normalized_keyword)
    return keywords

def filter_papers_by_keywords(
    papers: List[Dict],
    keywords: List[str],
) -> Tuple[List[Dict], Dict[str, int]]:
    """Filter papers by OR-matching keywords in title and abstract."""
    keyword_hit_counts = {keyword: 0 for keyword in keywords}
    if not keywords:
        return list(papers), keyword_hit_counts

    normalized_keywords = [
        (keyword, keyword.casefold()) for keyword in keywords
    ]
    filtered_papers = []

    for paper in papers:
        title = str(paper.get("title") or "")
        summary = str(paper.get("summary") or "")
        searchable_text = f"{title} {summary}".casefold()
        matched = False

        for keyword, normalized_keyword in normalized_keywords:
            if normalized_keyword in searchable_text:
                keyword_hit_counts[keyword] += 1
                matched = True

        if matched:
            filtered_papers.append(paper)

    return filtered_papers, keyword_hit_counts

def parse_max_ai_papers(raw_limit: Optional[str]) -> int:
    """Parse the optional per-run AI cap, falling back safely to 20."""
    if raw_limit is None or not raw_limit.strip():
        return MAX_AI_PAPERS_PER_RUN

    try:
        limit = int(raw_limit.strip())
    except (TypeError, ValueError):
        print(
            f"WARN: invalid MAX_AI_PAPERS_PER_RUN={raw_limit!r}; "
            f"using {MAX_AI_PAPERS_PER_RUN}",
            file=sys.stderr,
        )
        return MAX_AI_PAPERS_PER_RUN

    if limit <= 0:
        print(
            f"WARN: MAX_AI_PAPERS_PER_RUN must be positive; "
            f"using {MAX_AI_PAPERS_PER_RUN}",
            file=sys.stderr,
        )
        return MAX_AI_PAPERS_PER_RUN
    return limit

def apply_ai_paper_cap(
    papers: List[Dict],
    limit: int,
) -> Tuple[List[Dict], List[Dict]]:
    """Apply journal quality-first plus original-order exploration selection."""
    return select_journal_quality_and_exploration(papers, limit)

def process_single_item(chain, item: Dict, language: str) -> Dict:
    def check_github_code(content: str) -> Dict:
        """提取并验证 GitHub 链接"""
        code_info = {}

        # 1. 优先匹配 github.com/owner/repo 格式
        github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
        match = re.search(github_pattern, content)
        
        if match:
            owner, repo = match.groups()
            # 清理 repo 名称，去掉可能的 .git 后缀或末尾的标点
            repo = repo.rstrip(".git").rstrip(".,)")
            
            full_url = f"https://github.com/{owner}/{repo}"
            code_info["code_url"] = full_url
            
            # 尝试调用 GitHub API 获取信息
            github_token = os.environ.get("TOKEN_GITHUB")
            headers = {"Accept": "application/vnd.github.v3+json"}
            if github_token:
                headers["Authorization"] = f"token {github_token}"
            
            try:
                api_url = f"https://api.github.com/repos/{owner}/{repo}"
                resp = requests.get(api_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    code_info["code_stars"] = data.get("stargazers_count", 0)
                    code_info["code_last_update"] = data.get("pushed_at", "")[:10]
            except Exception:
                # API 调用失败不影响主流程
                pass
            return code_info

        # 2. 如果没有 github.com，尝试匹配 github.io
        github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
        match_io = re.search(github_io_pattern, content)
        
        if match_io:
            url = match_io.group(0)
            # 清理末尾标点
            url = url.rstrip(".,)")
            code_info["code_url"] = url
            # github.io 不进行 star 和 update 判断
                
        return code_info

    # 检测代码可用性
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    """处理单个数据项"""
    # Default structure with meaningful fallback values
    default_ai_fields = {
        "abstract_translation": "",
        "tldr": "摘要分析暂不可用",
        "motivation": "研究问题与理论背景分析暂不可用",
        "method": "数据、样本与研究方法分析暂不可用",
        "result": "研究结果分析暂不可用",
        "conclusion": "研究贡献与启示分析暂不可用"
    }
    
    try:
        response: Structure = chain.invoke({
            "language": language,
            "title": item.get("title") or "",
            "content": item.get("summary") or ""
        })
        item['AI'] = response.model_dump()
    except langchain_core.exceptions.OutputParserException as e:
        # 尝试从错误信息中提取 JSON 字符串并修复
        error_msg = str(e)
        partial_data = {}
        
        if "Function Structure arguments:" in error_msg:
            try:
                # 提取 JSON 字符串
                json_str = error_msg.split("Function Structure arguments:", 1)[1].strip().split('are not valid JSON')[0].strip()
                # 预处理 LaTeX 数学符号 - 使用四个反斜杠来确保正确转义
                json_str = json_str.replace('\\', '\\\\')
                # 尝试解析修复后的 JSON
                partial_data = json.loads(json_str)
            except Exception as json_e:
                print(f"Failed to parse JSON for {item.get('id', 'unknown')}: {json_e}", file=sys.stderr)
        
        # Merge partial data with defaults to ensure all fields exist
        item['AI'] = {**default_ai_fields, **partial_data}
        print(f"Using partial AI data for {item.get('id', 'unknown')}: {list(partial_data.keys())}", file=sys.stderr)
    except Exception as e:
        # Catch any other exceptions and provide default values
        print(f"Unexpected error for {item.get('id', 'unknown')}: {e}", file=sys.stderr)
        item['AI'] = default_ai_fields
    
    # Final validation to ensure all required fields exist
    for field in default_ai_fields.keys():
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    """并行处理所有数据项"""
    llm = ChatOpenAI(
            model=model_name,
            model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}
        ).with_structured_output(Structure, method="function_calling")

    print('Connect to:', model_name, file=sys.stderr)
    
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm
    
    # 使用线程池并行处理
    processed_data = [None] * len(data)  # 预分配结果列表
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_idx = {
            executor.submit(process_single_item, chain, item, language): idx
            for idx, item in enumerate(data)
        }
        
        # 使用tqdm显示进度
        for future in tqdm(
            as_completed(future_to_idx),
            total=len(data),
            desc="Processing items"
        ):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                print(f"Item at index {idx} generated an exception: {e}", file=sys.stderr)
                # Add default AI fields to ensure consistency
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "abstract_translation": "",
                    "tldr": "摘要分析暂不可用",
                    "motivation": "研究问题与理论背景分析暂不可用",
                    "method": "数据、样本与研究方法分析暂不可用",
                    "result": "研究结果分析暂不可用",
                    "conclusion": "研究贡献与启示分析暂不可用"
                }
    
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    # 检查并删除目标文件
    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f'Removed existing file: {target_file}', file=sys.stderr)

    # 读取数据
    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    print(
        f"去重后输入论文数量: {len(data)} / "
        f"Papers after deduplication: {len(data)}",
        file=sys.stderr,
    )

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    print('Open:', args.data, file=sys.stderr)

    keywords = parse_filter_keywords(os.environ.get("FILTER_KEYWORDS"))
    filtered_data, keyword_hit_counts = filter_papers_by_keywords(data, keywords)

    print(
        f"配置的有效关键词数量: {len(keywords)} / "
        f"Valid keyword count: {len(keywords)}",
        file=sys.stderr,
    )
    print(
        f"筛选后唯一论文数量: {len(filtered_data)} / "
        f"Unique papers after keyword filtering: {len(filtered_data)}",
        file=sys.stderr,
    )
    if keywords:
        for keyword in keywords:
            print(
                f"关键词命中数量 [{keyword}]: {keyword_hit_counts[keyword]} / "
                f"Hits for keyword [{keyword}]: {keyword_hit_counts[keyword]}",
                file=sys.stderr,
            )
    else:
        print(
            "未配置有效关键词，全部论文放行 / "
            "No valid keywords configured; all papers passed",
            file=sys.stderr,
        )

    max_ai_papers = parse_max_ai_papers(os.environ.get("MAX_AI_PAPERS_PER_RUN"))
    annotated_filtered_data = [annotate_paper_journal_rank(item) for item in filtered_data]
    selected_data, deferred_data = apply_ai_paper_cap(annotated_filtered_data, max_ai_papers)
    tier_counts = journal_tier_distribution(annotated_filtered_data)
    quality_slots, exploration_slots = selection_slots(max_ai_papers)
    quality_selected = [
        item for item in selected_data if item.get("selection_reason") == "journal_quality"
    ]
    exploration_selected = [
        item for item in selected_data if item.get("selection_reason") == "exploration"
    ]
    print(
        f"FILTER_KEYWORDS matched: {len(filtered_data)} / "
        f"keyword_matched_count={len(filtered_data)}",
        file=sys.stderr,
    )
    print("Journal tier distribution:", file=sys.stderr)
    for tier in ("S", "A", "B", "C", "U"):
        print(f"{tier}: {tier_counts[tier]}", file=sys.stderr)
    print(f"AI paper cap: {max_ai_papers} / ai_cap={max_ai_papers}", file=sys.stderr)
    print(f"Quality slots: {quality_slots}", file=sys.stderr)
    print(f"Exploration slots: {exploration_slots}", file=sys.stderr)
    print("Quality selected:", file=sys.stderr)
    for tier in ("S", "A", "B", "C", "U"):
        print(
            f"{tier}: {sum(1 for item in quality_selected if item.get('journal_tier') == tier)}",
            file=sys.stderr,
        )
    print(f"Exploration selected: {len(exploration_selected)}", file=sys.stderr)
    print(
        f"AI papers selected: {len(selected_data)} / "
        f"ai_selected_count={len(selected_data)}",
        file=sys.stderr,
    )
    print(
        f"AI papers deferred by cap: {len(deferred_data)} / "
        f"ai_deferred_count={len(deferred_data)}",
        file=sys.stderr,
    )
    for rank, item in enumerate(selected_data, start=1):
        print(
            "Selected paper: "
            f"rank={rank} tier={item.get('journal_tier', 'U')} "
            f"journal={item.get('journal') or item.get('source_name') or '<none>'} "
            f"title={item.get('title') or '<untitled>'} "
            f"selection_reason={item.get('selection_reason', '')}",
            file=sys.stderr,
        )

    if not selected_data:
        with open(target_file, "w", encoding="utf-8"):
            pass
        print(
            f"没有论文命中关键词，已生成空结果文件: {target_file} / "
            f"No AI papers selected; created empty result file: {target_file}",
            file=sys.stderr,
        )
        return

    # 并行处理所有数据
    processed_data = process_all_items(
        selected_data,
        model_name,
        language,
        args.max_workers
    )
    
    # 保存结果
    with open(target_file, "w", encoding="utf-8") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
