import re            # 이메일/전화번호 등 패턴 탐지
import unicodedata   # NFKC 등 텍스트 정규화
import hashlib       # PII 해시(익명화)
import uuid 

# =========================
# PII 필터 + 전처리 (보완판)
# =========================

# --- PII 패턴들 ---
# 주민등록번호: 캡처 그룹 + 체크섬 검증으로 FP 감소
RRN_RE   = re.compile(r"\b(\d{6})[- ]?(\d{7})\b")

# 신용카드 후보(룬 체크로 확정)
CARD_RE  = re.compile(r"\b(?:\d[ -]*?){13,19}\b")

# 계좌번호 후보: 하이픈/공백 허용으로 완화(후처리에서 길이 검증)
# - 연속 숫자/공백/하이픈으로 10~20 '토큰'을 허용하고, 양끝이 숫자로 더 이어지지 않도록 가드
ACC_RE   = re.compile(r"(?<!\d)(?:\d[ -]?){10,20}(?!\d)")
# 컨텍스트 키워드(있을 때만 계좌로 간주)
ACC_CTX  = re.compile(
    r"(계좌|입금|송금|이체|무통장|bank|account|농협|국민|신한|우리|하나|"
    r"카카오|토스|케이뱅크|ibk|기업|수협|새마을|우체국)", re.IGNORECASE
)

# 도로명 주소 간단 패턴 + 컨텍스트 키워드 동시 매칭 시에만 마스킹
ROAD_RE  = re.compile(
    r"\b[가-힣0-9A-Za-z]+(?:로|길|대로)\s?\d+(?:-\d+)?(?:\s?\d+호|\s?\d+층)?\b"
)
ROAD_CTX = re.compile(
    r"(주소|도로명|배달|택배|배송|거주|거주지|집|사무실|건물|아파트|빌라|오피스텔|"
    r"호\b|동\b|층\b|번지|우편번호)"
)

def _luhn_ok(num_str: str) -> bool:
    """신용카드 번호 룬(Luhn) 체크: 하이픈/공백 제거 후 검증"""
    digits = [int(c) for c in re.sub(r"\D", "", num_str)]
    if not (13 <= len(digits) <= 19):
        return False
    s = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d = d * 2
            if d > 9:
                d -= 9
        s += d
    return s % 10 == 0

def _rrn_ok(six: str, seven: str) -> bool:
    """주민등록번호 체크섬 검증 (YYMMDD-ABCDEFG, 마지막 G는 검증숫자)"""
    try:
        nums = [int(c) for c in six + seven]
        if len(nums) != 13:
            return False
        weights = [2,3,4,5,6,7,8,9,2,3,4,5]
        s = sum(a*b for a, b in zip(nums[:12], weights))
        check = (11 - (s % 11)) % 10
        return check == nums[12]
    except Exception:
        return False

def _mask_accounts(text: str) -> tuple[str, bool]:
    """
    계좌 마스킹:
      - 컨텍스트(ACC_CTX) 있을 때만 후보 탐색
      - 후보에서 비숫자 제거 후 길이 10~14면 계좌로 간주해 치환
    """
    if not ACC_CTX.search(text):
        return text, False

    changed = False

    def repl(m: re.Match) -> str:
        nonlocal changed
        raw = m.group()
        digits = re.sub(r"\D", "", raw)
        # 카드/주민번호/전화번호 등과 구분: 계좌는 보통 10~14자리
        if 10 <= len(digits) <= 14:
            changed = True
            return "[REDACTED:ACCOUNT]"
        return raw

    new_text = ACC_RE.sub(repl, text)
    return new_text, changed

def moderate_text(text: str):
    """
    반환: (action, masked_text, reasons)
    - action: "block" | "allow" | "allow_masked"
    - reasons: 탐지 사유 코드 리스트
    정책(개선):
      * 주민등록번호: 체크섬 통과 시 block (형식만 맞으면 X)
      * 신용카드: Luhn 통과 시 block
      * 계좌번호: 숫자패턴(하이픈/공백 허용) + 컨텍스트 동시 매칭 시 mask(길이 10~14)
      * 도로명주소: 주소패턴 + 컨텍스트 동시 매칭 시 mask
    """
    reasons: List[str] = []

    # 하드 차단: 주민등록번호(체크섬까지 통과해야 block)
    m_rrn = RRN_RE.search(text)
    if m_rrn and _rrn_ok(m_rrn.group(1), m_rrn.group(2)):
        return "block", text, ["resident_id"]

    # 하드 차단: 신용카드(룬 통과 시에만)
    for m in CARD_RE.finditer(text):
        if _luhn_ok(m.group()):
            return "block", text, ["credit_card"]

    # 마스킹 대상(컨텍스트 기반)
    masked = text
    before = masked

    # 계좌: 숫자 후보 + 컨텍스트 + 길이 검증
    masked, acc_changed = _mask_accounts(masked)
    if acc_changed:
        reasons.append("bank_account")

    # 도로명: 주소 패턴 + 컨텍스트 키워드 동시 매칭
    if ROAD_RE.search(masked) and ROAD_CTX.search(masked):
        masked = ROAD_RE.sub("[REDACTED:ROAD]", masked)
        reasons.append("road_address")

    action = "allow_masked" if masked != before else "allow"
    return action, masked, reasons

# --- 전처리 (URL/제로폭/공백/소문자/NFKC) ---

_url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# 제로폭/비가시 공백 문자 세트 확장: ZWSP/ZWJ/ZWNJ/WORD JOINER/BOM/NBSP 등
ZW_RE = re.compile(r"[\u200B-\u200D\u2060\uFEFF\u00A0\u202F]")

def preprocess_text(text: str) -> str:
    """URL/제로폭/공백 정리 + lower + NFKC 정규화"""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _url_re.sub(" ", t)
    t = ZW_RE.sub("", t)                 # 제로폭/비가시 공백 제거(기존 \u200b만 제거 → 확장)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t

def moderate_then_preprocess(raw_text: str):
    """
    1) PII 필터 (block 또는 마스킹)
    2) 전처리 적용
    반환: (action, preprocessed_text, reasons)
    """
    action, masked, reasons = moderate_text(raw_text)
    if action == "block":
        # 그대로 반려하고, 이후 파이프라인에 넘기지 않음
        return action, "", reasons
    clean = preprocess_text(masked)
    return action, clean, reasons

# (선택) 감사 로그: 원문 저장 금지, 마스킹 텍스트 해시만 저장
def log_moderation_event(action: str, reasons: List[str], masked_text: str, sink_path: str = "moderation.log") -> str:
    event = {
        "event_id": str(uuid.uuid4()),
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "action": action,                       # "block" | "allow" | "allow_masked"
        "reasons": reasons,                     # ["resident_id", ...]
        "masked_hash": hashlib.sha256(masked_text.encode("utf-8")).hexdigest() if masked_text else "",
        "masked_preview": masked_text[:120] if masked_text else "",
    }
    try:
        with open(sink_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        pass
    return event["event_id"]
