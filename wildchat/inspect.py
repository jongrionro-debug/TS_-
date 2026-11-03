#!/usr/bin/env python3
"""Aggregate WildChat Parquet metadata without exporting conversation text."""
import sys
from pathlib import Path
# Avoid shadowing the standard-library inspect module when invoked as a script.
sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import sqlite3
import tempfile

import pyarrow.parquet as pq
import tiktoken

REVISION = '7d6490e462285cf85d91eabea0f9a954fbddcd1f'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(',', ':')).encode()).digest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='Parquet file or recursively searched directory')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--max-rows', type=int, help='deterministic first N rows, sorted files; no random seed')
    args = p.parse_args()
    if args.max_rows is not None and args.max_rows <= 0:
        p.error('--max-rows must be positive')
    files = [args.input] if args.input.is_file() and args.input.suffix == '.parquet' else sorted(args.input.rglob('*.parquet')) if args.input.is_dir() else []
    if not files:
        p.error('input must contain at least one Parquet file')
    enc = tiktoken.get_encoding('o200k_base')
    counts, nulls, roles, msgnulls, exclusions = (Counter() for _ in range(5))
    schemas = set()
    with tempfile.TemporaryDirectory(prefix='wildchat-audit-') as tmp:
        db = sqlite3.connect(str(Path(tmp) / 'hashes.sqlite'))
        db.execute('CREATE TABLE hashes (kind TEXT, hash BLOB, n INTEGER, PRIMARY KEY(kind, hash)) WITHOUT ROWID')
        def add(kind, value):
            db.execute('INSERT INTO hashes VALUES (?, ?, 1) ON CONFLICT(kind, hash) DO UPDATE SET n=n+1', (kind, value))
        stop = False
        for path in files:
            pf = pq.ParquetFile(path)
            schemas.add(str(pf.schema_arrow.remove_metadata()))
            for batch in pf.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    if args.max_rows is not None and counts['rows'] >= args.max_rows:
                        stop = True
                        break
                    counts['rows'] += 1
                    for key, val in row.items():
                        if val is None: nulls[key] += 1
                    h = row.get('conversation_hash')
                    if h: add('id', digest(h))
                    add('row', digest(row))
                    messages = row.get('conversation') or []
                    add('conversation', digest([(m.get('role'), m.get('content')) if m else None for m in messages]))
                    prefix = []
                    for m in messages:
                        if not isinstance(m, dict):
                            counts['null_messages'] += 1
                            continue
                        counts['messages'] += 1
                        role = m.get('role') or '(null)'
                        roles[role] += 1
                        if m.get('timestamp') is not None: counts['timestamp_role_' + role] += 1
                        for key, val in m.items():
                            if val is None: msgnulls[key] += 1
                        if m.get('timestamp') is not None:
                            counts['message_timestamp_present'] += 1
                            if m.get('timestamp') == row.get('timestamp'): counts['message_timestamp_equals_row'] += 1
                        prefix.append((m.get('role'), m.get('content')))
                    # One prefix per row: first user prompt, exact visible content and role.
                    if messages and isinstance(messages[0], dict) and messages[0].get('role') == 'user' and messages[0].get('content'):
                        add('prefix', digest([(messages[0].get('role'), messages[0].get('content'))]))
                    if row.get('turn') == len(messages): counts['turn_equals_messages'] += 1
                    if row.get('turn') == sum(isinstance(m, dict) and m.get('role') == 'user' for m in messages): counts['turn_equals_user'] += 1
                    if row.get('turn') == sum(isinstance(m, dict) and m.get('role') == 'assistant' for m in messages): counts['turn_equals_assistant'] += 1
                    for i, m in enumerate(messages):
                        if not isinstance(m, dict) or m.get('role') != 'user': continue
                        nxt = messages[i+1] if i+1 < len(messages) else None
                        a = m.get('content')
                        if a is None: reason = 'user content 결측'
                        elif not isinstance(a, str) or not a.strip(): reason = 'user content 빈 문자열/공백'
                        elif not isinstance(nxt, dict) or nxt.get('role') != 'assistant': reason = '바로 다음 assistant 없음'
                        elif nxt.get('content') is None: reason = 'assistant content 결측'
                        elif not isinstance(nxt.get('content'), str) or not nxt['content'].strip(): reason = 'assistant content 빈 문자열/공백'
                        else:
                            counts['valid_pairs'] += 1
                            counts['input_tokens'] += len(enc.encode(a, disallowed_special=()))
                            counts['output_tokens'] += len(enc.encode(nxt['content'], disallowed_special=()))
                            continue
                        exclusions[reason] += 1
                db.commit()
                if stop: break
            if stop: break
        def stats(kind):
            unique, groups, repeated = db.execute('SELECT COUNT(*), COALESCE(SUM(n > 1),0), COALESCE(SUM(CASE WHEN n > 1 THEN n ELSE 0 END),0) FROM hashes WHERE kind=?', (kind,)).fetchone()
            return f'고유 {unique:,}, 반복 그룹 {groups:,}, 반복 그룹의 행 {repeated:,}, 추가 사본 {repeated-groups:,}'
        def table(counter, keys=None):
            return '\n'.join(f'| {str(k).replace("|", "/")} | {counter[k]:,} |' for k in (keys if keys is not None else sorted(counter))) or '| 없음 | 0 |'
        row_fields = sorted(pq.ParquetFile(files[0]).schema_arrow.names)
        lines = [
            '# WildChat 데이터 적합성 감사', '',
            '## 출처와 실행 범위', '',
            f'- 출처: [AllenAI WildChat-1M 데이터 카드](https://huggingface.co/datasets/allenai/WildChat-1M), Zhao 등, [WildChat 논문](https://arxiv.org/abs/2405.01470). 데이터 카드의 라이선스: [ODC-BY 1.0](https://opendatacommons.org/licenses/by/1-0/). 원 출처와 라이선스를 표시한다.',
            f'- 지정 데이터 리비전: `{REVISION}`. 리비전은 사용자가 제공한 출처 정보이며 파일 자체에서 검증할 수 없다.',
            f'- 입력 파일 {len(files):,}개, 처리 행 {counts["rows"]:,}개; 파일명: '+', '.join(f'`{x.name}`' for x in files)+'.',
            '- 선택: 입력 파일을 이름순으로 모두 순회' + (f'하되 앞 {args.max_rows:,}행으로 제한' if args.max_rows else '') + '. 무작위 추출 아님; 시드 없음. 일부 샤드만 입력했다면 전체 데이터 결과가 아니다.',
            ('- 이번 입력은 14개 샤드 중 첫 샤드 하나이며 전체 데이터 결과가 아니다.' if len(files) == 1 and files[0].name == 'train-00000-of-00014.parquet' else '- 결과는 지정한 입력 파일의 범위에 한정된다.'),
            '- 파일은 배치로 읽고 중복 집계는 임시 SQLite에 SHA-256 해시만 저장한다. 해시 충돌 가능성은 이론적으로 존재한다.',
            f'- 실행 버전: datasets {importlib.metadata.version("datasets")}, pyarrow {importlib.metadata.version("pyarrow")}, tiktoken {importlib.metadata.version("tiktoken")}; 토크나이저 o200k_base.',
            f'- 발견한 서로 다른 Parquet 스키마 수: {len(schemas)}. 행 단위 결측 표는 첫 파일의 스키마 기준이다.', '',
            '## 필드와 결측', '',
            '행은 대화 기록 하나이며 `conversation`은 메시지 목록이다. 데이터 카드는 `conversation_hash`를 대화 내용 해시로 설명하며 같은 내용을 가진 대화에서 중복될 수 있어 고유 키가 아니라고 명시한다. `model`, `turn`, `timestamp`, `language`, moderation, `toxic`, `redacted`와 지역·헤더 필드는 행 메타데이터다. 메시지는 `role`, `content`, `timestamp`, `turn_identifier`, 언어·지역·안전성 및 헤더를 갖는다. 결측은 null만 세며 빈 문자열은 별도다.', '',
            '| 행 필드 | null 행 수 |', '|---|---:|', table(nulls, row_fields), '',
            f'메시지 객체 {counts["messages"]:,}개, null 메시지 {counts["null_messages"]:,}개.', '',
            '| 메시지 역할 | 메시지 수 |', '|---|---:|', table(roles), '',
            '| 메시지 필드 | null 메시지 수 |', '|---|---:|', table(msgnulls, ['content','role','timestamp','turn_identifier','language','country','state','hashed_ip','header','redacted','toxic']), '',
            '## 중복·대화 구조', '',
            f'- 비어 있지 않은 conversation_hash: {stats("id")}. 동일 hash가 같은 실제 사용자 또는 독립 세션임을 보장하지 않는다.',
            f'- 모든 필드가 같은 행: {stats("row")}.',
            f'- 역할·가시 content 전체 목록이 같은 대화: {stats("conversation")}; 메타데이터는 무시.',
            f'- 첫 메시지가 비어 있지 않은 user일 때 첫 user 역할·텍스트가 정확히 같은 prefix: {stats("prefix")}. 이는 공유 prefix의 하한으로, 다른 시작 역할·부분 문자열·정규화·의미상 유사는 다루지 않는다.',
            f'- 행의 turn이 메시지 수와 일치 {counts["turn_equals_messages"]:,}행, user 수와 일치 {counts["turn_equals_user"]:,}행, assistant 수와 일치 {counts["turn_equals_assistant"]:,}행. 데이터 카드는 turn을 user와 assistant의 한 차례 상호작용 횟수로 정의한다. 관측 비교가 모든 예외의 생성 과정을 증명하지는 않는다.', '',
            '## 유효 쌍과 제외', '',
            '각 user 메시지를 후보로 삼고 바로 다음 메시지가 assistant이며 양쪽 content가 null도 공백만도 아닐 때 유효하다. 제외는 아래 순서의 첫 사유 하나로만 집계한다. 빈 문자열 판별 외의 텍스트 정규화는 하지 않는다. 행 간 중복은 제거하지 않으므로 같은 쌍의 반복이 포함된다.', '',
            f'- 유효 user→assistant 쌍 {counts["valid_pairs"]:,}개; user 후보 {roles["user"]:,}개.',
            f'- 유효 쌍의 현재 user 텍스트 입력 토큰 합 {counts["input_tokens"]:,}, assistant 텍스트 출력 토큰 합 {counts["output_tokens"]:,}. 각각 가시 텍스트만 o200k_base로 별도 계수한 재구성값이며 제공자 사용량이 아니다.', '',
            '| 제외 사유 (우선순위순) | 후보 수 |', '|---|---:|', table(exclusions, ['user content 결측','user content 빈 문자열/공백','바로 다음 assistant 없음','assistant content 결측','assistant content 빈 문자열/공백']), '',
            '## 시각과 후속 실험의 한계', '',
            f'- 역할별 timestamp 비결측: user {counts["timestamp_role_user"]:,}개, assistant {counts["timestamp_role_assistant"]:,}개.',
            f'- 메시지 timestamp 비결측 {counts["message_timestamp_present"]:,}개 중 행 timestamp와 동일 {counts["message_timestamp_equals_row"]:,}개. 데이터 카드는 행 timestamp를 마지막 턴의 UTC 시각, assistant 메시지 timestamp를 백엔드가 전체 응답을 수신한 시각이라고 설명한다. 사용자 요청 시각은 문서화되지 않았다. 메시지 timestamp의 정밀도·누락 원인 및 실제 요청 도착과의 관계는 확정할 수 없다.',
            '- 기록 시각을 각 user 요청 도착시각으로 가정하거나 턴 간격을 만들지 않는다. 중복·공유 prefix로 연결된 행을 같은 대화 그룹으로 묶고 그룹 기록시각 순으로 분할하는 방안은 전체 데이터에서 교차 그룹 누출과 시각 경계 처리를 검증한 뒤 확정한다. 결측 시각과 경계에 걸친 그룹의 처리도 후속 결정이 필요하다.',
            '- 집계는 관측 파일에서만 성립한다. 가시 텍스트의 삭제·마스킹, 선택 편향 및 토크나이저와 실제 API의 차이를 고려해야 한다. 학습·성능 평가는 수행하지 않았다.', '']
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text('\n'.join(lines), encoding='utf-8')
        print(f'Wrote {args.output}: {counts["rows"]:,} rows, {counts["valid_pairs"]:,} valid pairs')


if __name__ == '__main__':
    main()
