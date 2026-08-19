from dataclasses import dataclass, field

@dataclass(frozen=True)
class OfficialInstitution:
    name: str
    aliases: tuple[str, ...]
    official_domains: tuple[str, ...]
    official_phone_numbers: tuple[str, ...] = field(default_factory=tuple)
    known_sender_numbers: tuple[str, ...] = field(default_factory=tuple)


# 도메인/대표번호는 각 기관 공식 홈페이지를 직접 조회하거나(WebFetch) 검색으로 교차 확인해
# 채웠다 (2026-08 기준). 최초 작성 시 우리카드 도메인을 "uricard.com"으로 잘못 기재했었는데
# (실제로는 무관한 해외 서비스였음) 이 검증 과정에서 발견해 "wooricard.com"으로 수정했다 —
# 잘못된 공식 정보를 노출하면 오탐/오정보 제공으로 이어지므로, 향후 정보 추가/변경 시에도
# 반드시 공식 채널에서 직접 재확인할 것.
OFFICIAL_INSTITUTIONS: tuple[OfficialInstitution, ...] = (

    # 은행
    OfficialInstitution(
        "국민은행", ("국민은행", "KB국민은행"), ("kbstar.com",), ("1588-9999", "1599-9999", "1644-9999")
    ),
    OfficialInstitution("신한은행", ("신한은행",), ("shinhan.com",), ("1599-8000",)),
    OfficialInstitution("우리은행", ("우리은행",), ("wooribank.com",), ("1588-5000", "1599-5000", "1533-5000")),
    OfficialInstitution("하나은행", ("하나은행", "KEB하나은행"), ("kebhana.com",), ("1588-1111", "1599-1111")),
    OfficialInstitution(
        "NH농협은행", ("NH농협은행", "농협은행", "농협"), ("nonghyup.com",), ("1661-3000", "1661-2100")
    ),
    OfficialInstitution(
        "IBK기업은행", ("IBK기업은행", "기업은행"), ("ibk.co.kr",), ("1566-2566", "1588-2588")
    ),
    OfficialInstitution("카카오뱅크", ("카카오뱅크",), ("kakaobank.com",), ("1599-3333",)),
    OfficialInstitution("토스뱅크", ("토스뱅크",), ("tossbank.com",), ("1661-7654",)),
    OfficialInstitution("케이뱅크", ("케이뱅크",), ("kbanknow.com",), ("1522-1000",)),
    OfficialInstitution("새마을금고", ("새마을금고",), ("kfcc.co.kr",), ("1599-9000", "1588-8801")),
    OfficialInstitution("신협", ("신협",), ("cu.co.kr",), ("1566-6000", "1644-6000")),
    OfficialInstitution(
        "우체국", ("우체국", "우정사업본부"), ("epost.go.kr", "koreapost.go.kr"), ("1588-1300",)
    ),

    # 카드사
    OfficialInstitution("신한카드", ("신한카드",), ("shinhancard.com",), ("1544-7000",)),
    OfficialInstitution("삼성카드", ("삼성카드",), ("samsungcard.com",), ("1588-8700",)),
    OfficialInstitution("현대카드", ("현대카드",), ("hyundaicard.com",), ("1577-6000",)),
    OfficialInstitution("KB국민카드", ("KB국민카드", "국민카드"), ("kbcard.com",), ("1588-1688",)),
    OfficialInstitution("롯데카드", ("롯데카드",), ("lottecard.co.kr",), ("1588-8100",)),
    OfficialInstitution("우리카드", ("우리카드",), ("wooricard.com",), ("1588-9955", "1599-9955")),

    # 공공/사법기관
    OfficialInstitution("금융감독원", ("금융감독원",), ("fss.or.kr",), ("1332",)),
    OfficialInstitution("금융위원회", ("금융위원회",), ("fsc.go.kr",), ("02-2100-2500",)),
    OfficialInstitution("경찰청", ("경찰청",), ("police.go.kr",), ("182",)),
    OfficialInstitution("검찰청", ("검찰청",), ("spo.go.kr",), ("1301",)),

    # 홈택스(hometax.go.kr)는 국세청이 직접 운영하는 전자세정 서비스 도메인이라 함께 등록 
    OfficialInstitution("국세청", ("국세청",), ("nts.go.kr", "hometax.go.kr"), ("126",)),
    OfficialInstitution("관세청", ("관세청",), ("customs.go.kr",), ("125",)),
    OfficialInstitution(
        "국민건강보험공단",
        ("국민건강보험공단", "건강보험공단", "국민건강보험"),
        ("nhis.or.kr",),
        ("1577-1000",),
    ),
)
