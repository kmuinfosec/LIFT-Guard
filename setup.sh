#!/bin/bash
# 사내 TLS 프록시 환경에서 uv 가 pypi 인증서를 거부하므로 certifi 번들을 지정한다.
# 인증서 검증을 끄는 것이 아니라, 신뢰 저장소를 certifi 로 바꾸는 것이다.
set -e
cd "$(dirname "$0")"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$(python -c 'import certifi;print(certifi.where())')}"
uv sync --native-tls "$@"
echo "완료. 이후 실행: SSL_CERT_FILE=\"\$SSL_CERT_FILE\" uv run python scripts/01_asr.py"
