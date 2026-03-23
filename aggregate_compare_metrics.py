from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

from redraw_saved_trajectories import ModelEntry, collect_models

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = REPO_ROOT / 'compare_metrics_summary.xlsx'
DEFAULT_COMPARE_METHODS = ('iddpg', 'maddpg_nopf', 'maddpg_our_method')
DEFAULT_UAV_COUNTS = tuple(range(3, 9))
DEFAULT_TARGET_COUNTS = tuple(range(1, 6))
PRIMARY_METRICS = ('min_all_detect_step', 'total_detection_count', 'coverage_efficiency')
METRIC_DIRECTIONS = {
    'min_all_detect_step': 'min',
    'total_detection_count': 'max',
    'coverage_efficiency': 'max',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='汇总三类网络在不同无人机/目标数量下的最佳模型指标到 Excel。')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT_PATH, help='Excel 输出路径')
    parser.add_argument('--compare-methods', nargs='*', default=list(DEFAULT_COMPARE_METHODS), help='参与对比的方法名称')
    parser.add_argument('--uav-counts', nargs='*', type=int, default=list(DEFAULT_UAV_COUNTS), help='无人机数量列表')
    parser.add_argument('--target-counts', nargs='*', type=int, default=list(DEFAULT_TARGET_COUNTS), help='目标数量列表')
    return parser.parse_args()


def _score_value(metric: str, value: float) -> float:
    return -value if METRIC_DIRECTIONS[metric] == 'min' else value


def _normalize_metric_value(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_json_dict(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def normalize_method_name(model: ModelEntry) -> Optional[str]:
    if model.model_name.startswith('iddpg'):
        return 'iddpg'
    if model.model_name.startswith('maddpg_nopf'):
        return 'maddpg_nopf'
    if model.model_name.startswith('maddpg_pf'):
        return 'maddpg_our_method'
    return None


def discover_metrics_file(network_name: str, method: str, target_count: int) -> Optional[Path]:
    network_dir = REPO_ROOT / network_name
    if method == 'iddpg':
        candidate = network_dir / 'compare_results_100' / 'iddpg' / f'target_{target_count}' / 'results' / 'metrics_summary.json'
    elif method == 'maddpg_nopf':
        candidate = network_dir / 'compare_results_100' / 'maddpg_nopf' / f'target_{target_count}' / 'results' / 'metrics_summary.json'
    elif method == 'maddpg_our_method':
        candidate = network_dir / 'compare_results_100' / 'maddpg_our_method' / 'results' / f'target_{target_count}' / 'metrics_summary.json'
    else:
        raise KeyError(f'未配置方法: {method}')
    return candidate if candidate.exists() else None


def infer_metric_keys_from_model(model: ModelEntry) -> List[str]:
    keys: List[str] = []
    text_candidates = [model.model_name, model.model_path.name, model.model_path.as_posix()]

    for text in text_candidates:
        rank_match = re.search(r'rank[_-]?0*(\d+)', text)
        if rank_match:
            keys.append(f'rank_{int(rank_match.group(1)):02d}')

        top_ep_match = re.search(r'top\d+_ep[_-]?(\d+)', text)
        if top_ep_match:
            keys.append(f'top_ep_{int(top_ep_match.group(1))}')

        episode_match = re.search(r'episode[_-]?(\d+)', text)
        if episode_match:
            episode = int(episode_match.group(1))
            keys.append(f'episode_{episode}')
            keys.append(f'top_ep_{episode}')

        rank_ep_match = re.search(r'rank[_-]?0*(\d+)_ep[_-]?(\d+)', text)
        if rank_ep_match:
            keys.append(f'rank_{int(rank_ep_match.group(1)):02d}')
            keys.append(f'top_ep_{int(rank_ep_match.group(2))}')

    unique_keys: List[str] = []
    seen = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique_keys.append(key)
    return unique_keys


def discover_models_by_group(network_name: str, compare_methods: Iterable[str], target_counts: Iterable[int]) -> dict[tuple[str, int], List[ModelEntry]]:
    grouped: dict[tuple[str, int], List[ModelEntry]] = {}
    method_set = set(compare_methods)
    target_set = set(target_counts)
    for model in collect_models(network_name):
        method = normalize_method_name(model)
        if method is None or method not in method_set or model.n_target not in target_set:
            continue
        grouped.setdefault((method, model.n_target), []).append(model)
    return grouped


def pick_best_model(summary: dict, models: Iterable[ModelEntry]) -> Tuple[Optional[str], Optional[dict], List[str], str]:
    candidate_keys: List[str] = []
    key_to_models: dict[str, List[ModelEntry]] = {}
    for model in models:
        model_keys = infer_metric_keys_from_model(model)
        for key in model_keys:
            if key not in candidate_keys:
                candidate_keys.append(key)
            key_to_models.setdefault(key, []).append(model)

    available_keys = [key for key in candidate_keys if isinstance(summary.get(key), dict)]
    if not available_keys:
        return None, None, candidate_keys, 'no_collect_models_candidate_key_in_metrics_summary'

    best_key: Optional[str] = None
    best_metrics: Optional[dict] = None
    best_rank: Optional[Tuple[float, float, float, str]] = None
    for key in available_keys:
        metrics = summary[key]
        normalized = {
            metric: _normalize_metric_value(metrics.get(metric))
            for metric in PRIMARY_METRICS
        }
        if any(value is None for value in normalized.values()):
            continue
        rank = (
            _score_value('min_all_detect_step', normalized['min_all_detect_step']),
            _score_value('total_detection_count', normalized['total_detection_count']),
            _score_value('coverage_efficiency', normalized['coverage_efficiency']),
            key,
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_key = key
            best_metrics = normalized

    if best_key is None:
        return None, None, candidate_keys, 'collect_models_candidates_missing_required_metrics'
    return best_key, best_metrics, candidate_keys, ''


def build_summary_rows(compare_methods: Iterable[str], uav_counts: Iterable[int], target_counts: Iterable[int]) -> Tuple[List[dict], List[dict]]:
    rows: List[dict] = []
    missing_rows: List[dict] = []

    compare_methods = tuple(compare_methods)
    target_counts = tuple(target_counts)
    for uav_count in uav_counts:
        network_name = f'newnet_6_{uav_count}_compare'
        models_by_group = discover_models_by_group(network_name, compare_methods, target_counts)
        for method in compare_methods:
            for target_count in target_counts:
                metrics_path = discover_metrics_file(network_name, method, target_count)
                summary = load_json_dict(metrics_path) if metrics_path is not None else {}
                models = models_by_group.get((method, target_count), [])
                best_key, best_metrics, candidate_keys, missing_reason = pick_best_model(summary, models)
                row = {
                    'network': network_name,
                    'method': method,
                    'uav_count': int(uav_count),
                    'target_count': int(target_count),
                    'discovered_model_count': int(len(models)),
                    'best_model_key': best_key or '',
                    'candidate_model_keys': ', '.join(candidate_keys),
                    'metrics_file': '' if metrics_path is None else str(metrics_path),
                    'min_all_detect_step': '',
                    'total_detection_count': '',
                    'coverage_efficiency': '',
                }
                if best_metrics is not None:
                    for metric, value in best_metrics.items():
                        row[metric] = value
                else:
                    if not missing_reason:
                        missing_reason = 'metrics_summary_missing'
                    missing_rows.append(
                        {
                            'network': network_name,
                            'method': method,
                            'uav_count': int(uav_count),
                            'target_count': int(target_count),
                            'discovered_model_count': int(len(models)),
                            'candidate_model_keys': ', '.join(candidate_keys),
                            'metrics_file': '' if metrics_path is None else str(metrics_path),
                            'reason': missing_reason,
                        }
                    )
                rows.append(row)

    rows.sort(key=lambda item: (item['method'], item['uav_count'], item['target_count']))
    missing_rows.sort(key=lambda item: (item['method'], item['uav_count'], item['target_count']))
    return rows, missing_rows


def build_matrix_rows(rows: List[dict], metric: str, axis: str, axis_values: Iterable[int], compare_methods: Iterable[str], other_values: Iterable[int]) -> List[dict]:
    row_map = {
        (item['method'], item['uav_count'], item['target_count']): item
        for item in rows
    }
    matrix_rows: List[dict] = []
    for method in compare_methods:
        for axis_value in axis_values:
            output_row = {
                'method': method,
                axis: int(axis_value),
            }
            for other_value in other_values:
                if axis == 'uav_count':
                    item = row_map.get((method, int(axis_value), int(other_value)))
                    header = f'target_{int(other_value)}'
                else:
                    item = row_map.get((method, int(other_value), int(axis_value)))
                    header = f'uav_{int(other_value)}'
                value = '' if item is None else item.get(metric, '')
                output_row[header] = value
            matrix_rows.append(output_row)
    return matrix_rows


def col_name(index: int) -> str:
    result = ''
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def make_sheet_xml(headers: List[str], rows: List[dict]) -> str:
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        '  <sheetData>',
    ]

    header_cells = []
    for c, header in enumerate(headers):
        ref = f'{col_name(c)}1'
        header_cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(header))}</t></is></c>')
    lines.append('    <row r="1">' + ''.join(header_cells) + '</row>')

    for r, row in enumerate(rows, start=2):
        cells = []
        for c, header in enumerate(headers):
            ref = f'{col_name(c)}{r}'
            value = row.get(header, '')
            if isinstance(value, (int, float)):
                cells.append(f'<c r="{ref}"><v>{value}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        lines.append(f'    <row r="{r}">' + ''.join(cells) + '</row>')

    lines.extend(['  </sheetData>', '</worksheet>'])
    return '\n'.join(lines)


def write_xlsx(sheets: OrderedDict[str, Tuple[List[str], List[dict]]], output_path: Path) -> None:
    workbook_xml = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"',
        '          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
        '  <sheets>',
    ]
    workbook_rels = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
    ]

    for idx, name in enumerate(sheets.keys(), start=1):
        workbook_xml.append(f'    <sheet name="{escape(name)}" sheetId="{idx}" r:id="rId{idx}"/>')
        workbook_rels.append(
            f'  <Relationship Id="rId{idx}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )

    workbook_xml.extend(['  </sheets>', '</workbook>'])
    workbook_rels.extend(['</Relationships>'])

    content_types = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '  <Default Extension="xml" ContentType="application/xml"/>',
        '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for idx in range(1, len(sheets) + 1):
        content_types.append(
            f'  <Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types.append('</Types>')

    root_rels = '\n'.join([
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        '  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>',
        '</Relationships>',
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, 'w', compression=ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', '\n'.join(content_types))
        zf.writestr('_rels/.rels', root_rels)
        zf.writestr('xl/workbook.xml', '\n'.join(workbook_xml))
        zf.writestr('xl/_rels/workbook.xml.rels', '\n'.join(workbook_rels))
        for idx, (_name, (headers, sheet_rows)) in enumerate(sheets.items(), start=1):
            zf.writestr(f'xl/worksheets/sheet{idx}.xml', make_sheet_xml(headers, sheet_rows))


def main() -> None:
    args = parse_args()
    compare_methods = tuple(args.compare_methods)
    uav_counts = tuple(args.uav_counts)
    target_counts = tuple(args.target_counts)
    rows, missing_rows = build_summary_rows(compare_methods, uav_counts, target_counts)

    sheets: OrderedDict[str, Tuple[List[str], List[dict]]] = OrderedDict()
    summary_headers = [
        'network',
        'method',
        'uav_count',
        'target_count',
        'discovered_model_count',
        'best_model_key',
        'min_all_detect_step',
        'total_detection_count',
        'coverage_efficiency',
        'candidate_model_keys',
        'metrics_file',
    ]
    sheets['best_model_summary'] = (summary_headers, rows)

    for metric in PRIMARY_METRICS:
        uav_headers = ['method', 'uav_count'] + [f'target_{target}' for target in target_counts]
        target_headers = ['method', 'target_count'] + [f'uav_{uav}' for uav in uav_counts]
        sheets[f'{metric[:18]}_by_uav'] = (
            uav_headers,
            build_matrix_rows(rows, metric, 'uav_count', uav_counts, compare_methods, target_counts),
        )
        sheets[f'{metric[:15]}_by_tar'] = (
            target_headers,
            build_matrix_rows(rows, metric, 'target_count', target_counts, compare_methods, uav_counts),
        )

    missing_headers = ['network', 'method', 'uav_count', 'target_count', 'discovered_model_count', 'candidate_model_keys', 'metrics_file', 'reason']
    sheets['missing_entries'] = (missing_headers, missing_rows)

    write_xlsx(sheets, args.output)
    print(f'已生成 Excel: {args.output}')
    print(f'总组合数: {len(rows)}')
    print(f'缺失组合数: {len(missing_rows)}')


if __name__ == '__main__':
    main()
