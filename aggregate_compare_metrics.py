from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

BASE_DIR = Path('/workspace/wrjjq')
OUTPUT_PATH = Path('/home/wensheng/gjq_workspace/wrjjq/compare_metrics_summary.xlsx')

METHOD_PATHS = OrderedDict(
    {
        'iddpg': lambda compare_dir, target: compare_dir / 'iddpg' / f'target_{target}' / 'results' / 'metrics_summary.json',
        'maddpg_nopf': lambda compare_dir, target: compare_dir / 'maddpg_nopf' / f'target_{target}' / 'results' / 'metrics_summary.json',
        'maddpg_our_method': lambda compare_dir, target: compare_dir / 'maddpg_our_method' / 'results' / f'target_{target}' / 'metrics_summary.json',
    }
)

METRICS = [
    'min_all_detect_step',
    'total_detection_count',
    'overlap_rate',
    'collision_count',
    'coverage_efficiency',
]


def pick_best_entry(summary: dict) -> tuple[str, dict]:
    if 'rank_01' in summary:
        return 'rank_01', summary['rank_01']

    top_keys = [k for k in summary if k.startswith('top_ep_')]
    if top_keys:
        return top_keys[0], summary[top_keys[0]]

    first_key = next(iter(summary))
    return first_key, summary[first_key]


def build_tables() -> tuple[dict[str, list[dict]], list[dict]]:
    metric_tables: dict[str, list[dict]] = {metric: [] for metric in METRICS}
    best_model_rows: list[dict] = []

    for uav_count in range(3, 7):
        compare_dir = BASE_DIR / f'newnet_6_{uav_count}_compare' / 'compare_results'
        if not compare_dir.exists():
            raise FileNotFoundError(f'缺少目录: {compare_dir}')

        for target_count in range(1, 5):
            group_key = f'UAV_{uav_count}_Target_{target_count}'
            metric_row_map = {
                metric: {
                    'uav_count': uav_count,
                    'target_count': target_count,
                    'group': group_key,
                }
                for metric in METRICS
            }

            for method, path_builder in METHOD_PATHS.items():
                metrics_file = path_builder(compare_dir, target_count)
                if not metrics_file.exists():
                    raise FileNotFoundError(f'缺少指标文件: {metrics_file}')

                with metrics_file.open('r', encoding='utf-8') as f:
                    summary = json.load(f)

                if not isinstance(summary, dict) or not summary:
                    raise ValueError(f'指标文件为空或格式错误: {metrics_file}')

                best_key, best_metrics = pick_best_entry(summary)
                best_model_rows.append(
                    {
                        'uav_count': uav_count,
                        'target_count': target_count,
                        'method': method,
                        'best_model_key': best_key,
                        'source_file': str(metrics_file),
                    }
                )

                for metric in METRICS:
                    if metric not in best_metrics:
                        raise KeyError(f'{metrics_file} 的最佳模型 {best_key} 缺少指标 {metric}')
                    metric_row_map[metric][method] = best_metrics[metric]

            for metric in METRICS:
                metric_tables[metric].append(metric_row_map[metric])

    for rows in metric_tables.values():
        rows.sort(key=lambda x: (x['uav_count'], x['target_count']))
    best_model_rows.sort(key=lambda x: (x['uav_count'], x['target_count'], x['method']))

    return metric_tables, best_model_rows


def col_name(index: int) -> str:
    result = ''
    n = index + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result = chr(65 + rem) + result
    return result


def make_sheet_xml(headers: list[str], rows: list[dict]) -> str:
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


def write_xlsx(sheets: OrderedDict[str, tuple[list[str], list[dict]]], output_path: Path) -> None:
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

    with ZipFile(output_path, 'w', compression=ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', '\n'.join(content_types))
        zf.writestr('_rels/.rels', root_rels)
        zf.writestr('xl/workbook.xml', '\n'.join(workbook_xml))
        zf.writestr('xl/_rels/workbook.xml.rels', '\n'.join(workbook_rels))

        for idx, (_name, (headers, rows)) in enumerate(sheets.items(), start=1):
            zf.writestr(f'xl/worksheets/sheet{idx}.xml', make_sheet_xml(headers, rows))


def main() -> None:
    metric_tables, best_models = build_tables()

    sheets: OrderedDict[str, tuple[list[str], list[dict]]] = OrderedDict()
    metric_headers = ['uav_count', 'target_count', 'group', 'iddpg', 'maddpg_nopf', 'maddpg_our_method']
    for metric in METRICS:
        sheets[metric[:31]] = (metric_headers, metric_tables[metric])

    best_headers = ['uav_count', 'target_count', 'method', 'best_model_key', 'source_file']
    sheets['best_model_keys'] = (best_headers, best_models)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_xlsx(sheets, OUTPUT_PATH)

    print(f'已生成 Excel: {OUTPUT_PATH}')
    print('共处理对比组数量:', len(best_models) // len(METHOD_PATHS))


if __name__ == '__main__':
    main()
