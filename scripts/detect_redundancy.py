#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冗余描述检测脚本
检测小说章节中的重复描写、套路化表达、冗余描述。
输出 Markdown 格式的检测报告。
"""

import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 修复 Windows 控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


# ========== 内置套路词库 ==========

CRUTCH_PHRASES = {
    # 情绪反应套路
    "心中一震": "胃像被人攥了一把 / 后背窜过一道寒意 / 脚步顿住",
    "不由得": "[直接写动作，删除此词]",
    "不由自主": "[直接写动作，删除此词]",
    "嘴角微微上扬": "咧开嘴 / 眼尾皱纹加深 / 低头笑了一声",
    "嘴角上扬": "咧开嘴 / 鼻子哼了哼 / 嗤了一声",
    "嘴角勾起": "露出笑意 / 眉梢一挑",
    "眼中闪过": "眉头拧了一下 / 下意识攥紧衣角 / 呼吸停了半拍",
    "深吸一口气": "用力揉了把脸 / 把手插进头发里 / 灌了一口凉水",
    "心如刀绞": "胸口像压了块石头 / 鼻腔一阵酸涩 / 视线模糊了",
    "浑身一颤": "手抖了一下 / 膝盖发软 / 后退了半步",
    "目光如炬": "盯得人头皮发麻 / 那种眼神让人想后退",
    "心头一紧": "喉咙像被掐住 / 手心渗出了汗",
    "眉头紧锁": "眉心拧成一个结 / 额头的青筋跳了跳",
    "瞳孔微缩": "眼皮跳了一下 / 目光倏地锐利起来",
    "倒吸一口凉气": "后脊梁一阵发凉 / 头皮一麻",

    # 过渡语套路
    "就在这时": "[直接写发生的事]",
    "话说回来": "[用角色动作转场]",
    "与此同时": "[切视角直接开始]",
    "时光荏苒": "[用具体细节暗示时间]",
    "良久之后": "[用动作或场景变化代替]",
    "这一刻": "[删除，直接写]",
    "一时间": "[删除，直接写]",

    # AI味形容词堆砌
    "璀璨夺目": "[用具体光线效果代替]",
    "波澜壮阔": "[用具体场景代替]",
    "气势恢宏": "[用具体细节代替]",
    "美轮美奂": "[用具体视觉细节代替]",
    "如诗如画": "[删除，用具体描写代替]",
    "心潮澎湃": "心跳快了 / 手心攥出了汗 / 声音在抖",
    "热血沸腾": "血往头上涌 / 拳头握紧了 / 牙关咬死",

    # 描述堆砌标志
    "仿佛": "[如果一页出现3次以上，需要精简]",
    "似乎": "[如果一页出现3次以上，需要精简]",
    "宛如": "[如果一页出现3次以上，需要精简]",
    "犹如": "[如果一页出现3次以上，需要精简]",
}


def extract_chinese_text(file_path: Path) -> tuple:
    """提取章节文件的中文正文内容，返回(文件名, 正文)"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 跳过 Markdown 元数据
    lines = content.split('\n')
    content_start = 0
    for i, line in enumerate(lines):
        if line.startswith('#') and '章' in line:
            content_start = i + 1
            break

    main_content = '\n'.join(lines[content_start:])
    # 移除Markdown标记
    main_content = re.sub(r'#{1,6}\s*.*', '', main_content)
    main_content = re.sub(r'\*\*(.*?)\*\*', r'\1', main_content)
    main_content = re.sub(r'\*(.*?)\*', r'\1', main_content)
    main_content = re.sub(r'---+', '', main_content)

    return file_path.name, main_content


def find_chapter_files(directory: str) -> list:
    """查找目录下所有章节文件（支持卷目录递归）"""
    dir_path = Path(directory)
    chapter_files = []

    # 直接在目录下查找
    chapter_files.extend(sorted(dir_path.glob('第*.md')))

    # 在卷子目录下查找
    for sub_dir in sorted(dir_path.iterdir()):
        if sub_dir.is_dir() and ('卷' in sub_dir.name):
            chapter_files.extend(sorted(sub_dir.glob('第*.md')))

    return chapter_files


def detect_crutch_phrases(chapters: dict) -> list:
    """检测套路词使用频率"""
    results = []

    for phrase, suggestion in CRUTCH_PHRASES.items():
        total_count = 0
        locations = []

        for filename, content in chapters.items():
            count = content.count(phrase)
            if count > 0:
                total_count += count
                locations.append(f"{filename}({count}次)")

        if total_count >= 3:  # 出现3次以上才报告
            results.append({
                'phrase': phrase,
                'count': total_count,
                'locations': locations,
                'suggestion': suggestion,
            })

    results.sort(key=lambda x: x['count'], reverse=True)
    return results


def detect_ngram_repetition(chapters: dict, n: int = 6, threshold: int = 5) -> list:
    """检测跨章节重复的n-gram"""
    ngram_locations = defaultdict(list)

    for filename, content in chapters.items():
        # 提取纯中文字符序列
        chinese_only = re.findall(r'[\u4e00-\u9fff]+', content)
        text = ''.join(chinese_only)

        seen_in_file = set()
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            if ngram not in seen_in_file:
                seen_in_file.add(ngram)
                ngram_locations[ngram].append(filename)

    # 过滤：至少在 threshold 个不同章节中出现
    repeated = []
    for ngram, files in ngram_locations.items():
        unique_files = set(files)
        if len(unique_files) >= threshold:
            repeated.append({
                'ngram': ngram,
                'file_count': len(unique_files),
                'files': sorted(unique_files)[:5],  # 只展示前5个
            })

    repeated.sort(key=lambda x: x['file_count'], reverse=True)
    return repeated[:30]  # 只返回Top 30


def detect_similar_descriptions(chapters: dict, min_length: int = 10) -> list:
    """检测跨章节的相似长描述片段"""
    # 提取较长的中文短语（10+字符）
    phrase_locations = defaultdict(list)

    for filename, content in chapters.items():
        # 按标点符号分割成句子
        sentences = re.split(r'[。！？；\n]', content)
        for sentence in sentences:
            # 提取纯中文
            chinese = ''.join(re.findall(r'[\u4e00-\u9fff]', sentence))
            if len(chinese) >= min_length:
                # 用滑动窗口提取子串
                for i in range(len(chinese) - min_length + 1):
                    substr = chinese[i:i + min_length]
                    phrase_locations[substr].append(filename)

    # 找出在多个章节中出现的描述
    repeated = []
    seen_overlaps = set()

    for phrase, files in phrase_locations.items():
        unique_files = set(files)
        if len(unique_files) >= 3:
            # 避免报告高度重叠的子串
            is_overlap = False
            for seen in seen_overlaps:
                if phrase in seen or seen in phrase:
                    is_overlap = True
                    break
            if not is_overlap:
                seen_overlaps.add(phrase)
                repeated.append({
                    'phrase': phrase,
                    'file_count': len(unique_files),
                    'files': sorted(unique_files)[:5],
                })

    repeated.sort(key=lambda x: x['file_count'], reverse=True)
    return repeated[:20]


def detect_adjective_pileup(chapters: dict) -> list:
    """检测形容词堆砌（连续多个"的"修饰）"""
    pattern = re.compile(r'[\u4e00-\u9fff]{2,4}的[\u4e00-\u9fff]{2,4}的[\u4e00-\u9fff]{2,4}的')
    results = []

    for filename, content in chapters.items():
        matches = pattern.findall(content)
        if matches:
            results.append({
                'file': filename,
                'count': len(matches),
                'examples': matches[:3],
            })

    return results


def detect_punctuation_issues(chapters: dict) -> list:
    """检测标点符号问题"""
    results = []

    for filename, content in chapters.items():
        issues = []

        # 1. 英文标点混入（在中文上下文中）
        # 检测被中文字符包围的英文标点
        en_punct_pattern = re.compile(
            r'[\u4e00-\u9fff][,.:;!?()]\s*[\u4e00-\u9fff]'
        )
        en_matches = en_punct_pattern.findall(content)
        if en_matches:
            issues.append({
                'type': '英文标点混入',
                'count': len(en_matches),
                'examples': en_matches[:3],
            })

        # 2. 英文省略号（三个点代替六个点）
        en_ellipsis = re.findall(r'\.{3}', content)
        if en_ellipsis:
            issues.append({
                'type': '英文省略号（应为……）',
                'count': len(en_ellipsis),
                'examples': ['...'],
            })

        # 3. 英文破折号（--代替——）
        en_dash = re.findall(r'(?<!-)--(?!-)', content)
        if en_dash:
            issues.append({
                'type': '英文破折号（应为——）',
                'count': len(en_dash),
                'examples': ['--'],
            })

        # 4. 标点重复/堆砌
        punct_repeat = re.findall(r'[？！]{2,}', content)
        if punct_repeat:
            issues.append({
                'type': '标点重复堆砌',
                'count': len(punct_repeat),
                'examples': punct_repeat[:3],
            })

        # 5. 省略号过密（统计段落中的省略号）
        paragraphs = content.split('\n\n')
        dense_ellipsis = 0
        for para in paragraphs:
            ellipsis_count = para.count('……') + para.count('...')
            if ellipsis_count >= 3:
                dense_ellipsis += 1
        if dense_ellipsis > 0:
            issues.append({
                'type': '省略号过密（同段3个以上）',
                'count': dense_ellipsis,
                'examples': [f'{dense_ellipsis}个段落'],
            })

        # 6. 逗号过载（单句超过3个逗号）
        sentences = re.split(r'[。！？]', content)
        comma_overload = 0
        for sent in sentences:
            if sent.count('，') > 3:
                comma_overload += 1
        if comma_overload > 0:
            issues.append({
                'type': '逗号过载（单句超过3个逗号）',
                'count': comma_overload,
                'examples': [f'{comma_overload}个句子'],
            })

        if issues:
            results.append({
                'file': filename,
                'issues': issues,
            })

    return results


def detect_sentence_pattern_issues(chapters: dict) -> list:
    """检测句式单一性问题"""
    results = []

    for filename, content in chapters.items():
        issues = []

        # 按句号/感叹号/问号分句
        sentences = [s.strip() for s in re.split(r'[。！？]', content) if s.strip()]
        if len(sentences) < 10:
            continue

        # 1. 人称代词开头占比
        pronoun_starts = 0
        total_valid = 0
        for sent in sentences:
            # 取句子的第一个中文字符或前两个
            first_chars = ''.join(re.findall(r'[\u4e00-\u9fff]', sent[:6]))
            if not first_chars:
                continue
            total_valid += 1
            if first_chars and first_chars[0] in '他她我它们':
                pronoun_starts += 1

        if total_valid > 0:
            ratio = pronoun_starts / total_valid
            if ratio > 0.4:
                issues.append({
                    'type': '人称代词开头过多',
                    'detail': f'{pronoun_starts}/{total_valid}句（{ratio:.0%}），建议降至30%以下',
                    'severity': 'high',
                })
            elif ratio > 0.3:
                issues.append({
                    'type': '人称代词开头偏多',
                    'detail': f'{pronoun_starts}/{total_valid}句（{ratio:.0%}），建议降至30%以下',
                    'severity': 'medium',
                })

        # 2. 连续"了"结尾
        consecutive_le = 0
        max_consecutive_le = 0
        for sent in sentences:
            chinese_chars = re.findall(r'[\u4e00-\u9fff]', sent)
            if chinese_chars and chinese_chars[-1] == '了':
                consecutive_le += 1
                max_consecutive_le = max(max_consecutive_le, consecutive_le)
            else:
                consecutive_le = 0

        if max_consecutive_le >= 3:
            issues.append({
                'type': '连续"了"字结尾',
                'detail': f'最多连续{max_consecutive_le}句以"了"结尾',
                'severity': 'high' if max_consecutive_le >= 5 else 'medium',
            })

        # 3. 连续相同段首词
        paragraphs = [p.strip() for p in content.split('\n') if p.strip() and len(p.strip()) > 5]
        if len(paragraphs) >= 5:
            max_same_start = 1
            current_run = 1
            for i in range(1, len(paragraphs)):
                first_a = ''.join(re.findall(r'[\u4e00-\u9fff]', paragraphs[i - 1][:4]))
                first_b = ''.join(re.findall(r'[\u4e00-\u9fff]', paragraphs[i][:4]))
                if first_a and first_b and first_a[:2] == first_b[:2]:
                    current_run += 1
                    max_same_start = max(max_same_start, current_run)
                else:
                    current_run = 1

            if max_same_start >= 3:
                issues.append({
                    'type': '连续段落以相同词开头',
                    'detail': f'最多连续{max_same_start}段以相同词开头',
                    'severity': 'high' if max_same_start >= 4 else 'medium',
                })

        # 4. 句子长度单调（连续5句长度相近）
        sent_lengths = []
        for sent in sentences:
            chinese_len = len(re.findall(r'[\u4e00-\u9fff]', sent))
            if chinese_len > 0:
                sent_lengths.append(chinese_len)

        max_similar_run = 1
        current_run = 1
        for i in range(1, len(sent_lengths)):
            if abs(sent_lengths[i] - sent_lengths[i - 1]) <= 5:
                current_run += 1
                max_similar_run = max(max_similar_run, current_run)
            else:
                current_run = 1

        if max_similar_run >= 5:
            issues.append({
                'type': '句子长度单调',
                'detail': f'连续{max_similar_run}句长度相近（±5字），缺少长短交替',
                'severity': 'medium',
            })

        # 5. "然后/接着/随后"连接词过度使用
        connectors = ['然后', '接着', '随后', '之后', '紧接着']
        connector_count = 0
        for conn in connectors:
            connector_count += content.count(conn)
        # 按每千字计算密度
        total_chars = len(re.findall(r'[\u4e00-\u9fff]', content))
        if total_chars > 0:
            density = connector_count / (total_chars / 1000)
            if density > 3:
                issues.append({
                    'type': '顺序连接词过密',
                    'detail': f'"然后/接着/随后"等词共{connector_count}次（每千字{density:.1f}次），建议降至每千字2次以下',
                    'severity': 'medium',
                })

        if issues:
            results.append({
                'file': filename,
                'issues': issues,
            })

    return results


def generate_report(directory: str, crutch_results: list, ngram_results: list,
                    similar_results: list, pileup_results: list,
                    punct_results: list, pattern_results: list,
                    chapter_count: int, top_n: int = 20) -> str:
    """生成 Markdown 格式的检测报告"""
    lines = []
    lines.append("# 质量检测报告")
    lines.append("")
    lines.append(f"- **检测目录**：{directory}")
    lines.append(f"- **检测章节数**：{chapter_count}")
    lines.append("")

    # 总结
    total_issues = (len(crutch_results) + len(ngram_results) + len(similar_results)
                    + len(pileup_results) + len(punct_results) + len(pattern_results))
    if total_issues == 0:
        lines.append("未发现明显问题。")
        return '\n'.join(lines)

    lines.append(f"共发现 **{total_issues}** 类问题。")
    lines.append("")

    # 1. 套路词
    if crutch_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 套路化表达（{len(crutch_results)} 项）")
        lines.append("")
        lines.append("| 套路词 | 出现次数 | 出现位置 | 建议替代 |")
        lines.append("|--------|---------|---------|---------|")
        for r in crutch_results[:top_n]:
            locs = '、'.join(r['locations'][:3])
            if len(r['locations']) > 3:
                locs += f" 等{len(r['locations'])}处"
            lines.append(f"| {r['phrase']} | {r['count']} | {locs} | {r['suggestion']} |")
        lines.append("")

    # 2. 高频重复片段
    if ngram_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 高频重复片段（{len(ngram_results)} 项）")
        lines.append("")
        lines.append("以下6字以上的片段在多个不同章节中重复出现：")
        lines.append("")
        lines.append("| 重复片段 | 出现章节数 | 示例位置 |")
        lines.append("|---------|-----------|---------|")
        for r in ngram_results[:top_n]:
            files = '、'.join(r['files'][:3])
            if r['file_count'] > 3:
                files += f" 等{r['file_count']}章"
            lines.append(f"| {r['ngram']} | {r['file_count']} | {files} |")
        lines.append("")

    # 3. 相似描述
    if similar_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 跨章节相似描述（{len(similar_results)} 项）")
        lines.append("")
        lines.append("以下10字以上的描述片段在3个以上章节中出现：")
        lines.append("")
        lines.append("| 描述片段 | 出现章节数 | 示例位置 |")
        lines.append("|---------|-----------|---------|")
        for r in similar_results[:top_n]:
            files = '、'.join(r['files'][:3])
            if r['file_count'] > 3:
                files += f" 等{r['file_count']}章"
            lines.append(f"| {r['phrase']} | {r['file_count']} | {files} |")
        lines.append("")

    # 4. 形容词堆砌
    if pileup_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 形容词堆砌（{len(pileup_results)} 个章节）")
        lines.append("")
        lines.append('以下章节存在连续多个"的"修饰的堆砌结构：')
        lines.append("")
        for r in pileup_results[:top_n]:
            examples = '、'.join(f'"{e}"' for e in r['examples'])
            lines.append(f"- **{r['file']}**：{r['count']}处，如 {examples}")
        lines.append("")

    # 5. 标点符号问题
    if punct_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 标点符号问题（{len(punct_results)} 个章节）")
        lines.append("")
        for r in punct_results[:top_n]:
            lines.append(f"### {r['file']}")
            for issue in r['issues']:
                examples_str = '、'.join(str(e) for e in issue['examples'][:3])
                lines.append(f"- **{issue['type']}**：{issue['count']}处（如 {examples_str}）")
            lines.append("")

    # 6. 句式单一性问题
    if pattern_results:
        lines.append("---")
        lines.append("")
        lines.append(f"## 句式单一性问题（{len(pattern_results)} 个章节）")
        lines.append("")
        for r in pattern_results[:top_n]:
            lines.append(f"### {r['file']}")
            for issue in r['issues']:
                severity_icon = '!!!' if issue['severity'] == 'high' else '!'
                lines.append(f"- {severity_icon} **{issue['type']}**：{issue['detail']}")
            lines.append("")

    # 优化建议
    lines.append("---")
    lines.append("")
    lines.append("## 优化建议")
    lines.append("")
    lines.append("1. **冗余问题**：查阅 `references/redundancy-check.md`")
    lines.append("2. **标点问题**：查阅 `references/punctuation-guide.md`")
    lines.append("3. **句式问题**：查阅 `references/sentence-variety.md`")
    lines.append("")

    return '\n'.join(lines)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print('用法:')
        print('  python detect_redundancy.py <章节目录路径> [选项]')
        print('')
        print('选项:')
        print('  --threshold N    n-gram重复阈值，默认5（在N个以上章节中出现才报告）')
        print('  --top N          显示Top N条结果，默认20')
        print('  --output FILE    输出报告到文件（默认打印到终端）')
        print('')
        print('示例:')
        print('  python detect_redundancy.py novels/故事/')
        print('  python detect_redundancy.py novels/故事/卷一/')
        print('  python detect_redundancy.py novels/故事/ --threshold 3 --top 30')
        print('  python detect_redundancy.py novels/故事/ --output report.md')
        return

    directory = sys.argv[1]
    threshold = 5
    top_n = 20
    output_file = None

    # 解析可选参数
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--threshold' and i + 1 < len(sys.argv):
            threshold = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--top' and i + 1 < len(sys.argv):
            top_n = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            i += 1

    # 查找章节文件
    chapter_files = find_chapter_files(directory)
    if not chapter_files:
        print(f'错误: 在 {directory} 中未找到章节文件')
        return

    print(f'找到 {len(chapter_files)} 个章节文件，开始检测...')

    # 读取所有章节
    chapters = {}
    for f in chapter_files:
        name, content = extract_chinese_text(f)
        if content.strip():
            chapters[name] = content

    if not chapters:
        print('错误: 所有章节文件均无有效内容')
        return

    # 执行检测
    print('检测套路化表达...')
    crutch_results = detect_crutch_phrases(chapters)

    print('检测高频重复片段...')
    ngram_results = detect_ngram_repetition(chapters, n=6, threshold=threshold)

    print('检测跨章节相似描述...')
    similar_results = detect_similar_descriptions(chapters)

    print('检测形容词堆砌...')
    pileup_results = detect_adjective_pileup(chapters)

    print('检测标点符号问题...')
    punct_results = detect_punctuation_issues(chapters)

    print('检测句式单一性...')
    pattern_results = detect_sentence_pattern_issues(chapters)

    # 生成报告
    report = generate_report(directory, crutch_results, ngram_results,
                            similar_results, pileup_results,
                            punct_results, pattern_results,
                            len(chapters), top_n)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'\n报告已保存到: {output_file}')
    else:
        print('\n' + report)


if __name__ == '__main__':
    main()
