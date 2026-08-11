from dataclasses import dataclass, field


@dataclass(frozen=True)
class OfficialInstitution:
    # 공식 기관명 (로그/응답 표시용)
    name: str
    # 문자에서 실제 쓰이는 표기 변형 (이 중 하나라도 본문에 있으면 해당 기관 언급으로 간주)
    aliases: tuple[str, ...]
    # 공식 도메인. 서브도메인도 인정하려면 상위 도메인만 등록 (예: "kbstar.com"이면 "obank.kbstar.com"도 매치)
    official_domains: tuple[str, ...]
    # 공식 대표번호 / 자주 쓰이는 발신번호. 검증된 데이터가 확보되기 전까지는 비워둔다 —
    # 잘못된 번호를 "공식"으로 잘못 표기하면 사용자에게 잘못된 정보를 주는 위험이
    # 도메인 오류보다 크므로, 각 기관 공식 채널을 통한 재검증 없이는 채우지 않는다.
    official_phone_numbers: tuple[str, ...] = field(default_factory=tuple)
    known_sender_numbers: tuple[str, ...] = field(default_factory=tuple)


# NOTE(검증 필요): 아래 도메인은 공개적으로 널리 알려진 공식 도메인을 기준으로 작성했으나,
# 실서비스 반영 전 각 기관의 공식 발표 채널을 통해 최종 재검증이 필요하다.
# 전화번호 필드는 위와 같은 이유로 의도적으로 비워뒀다 — 검증된 데이터 확보 후 채울 것.
OFFICIAL_INSTITUTIONS: tuple[OfficialInstitution, ...] = (
    # 은행
    OfficialInstitution("국민은행", ("국민은행", "KB국민은행"), ("kbstar.com",)),
    OfficialInstitution("신한은행", ("신한은행",), ("shinhan.com",)),
    OfficialInstitution("우리은행", ("우리은행",), ("wooribank.com",)),
    OfficialInstitution("하나은행", ("하나은행", "KEB하나은행"), ("kebhana.com",)),
    OfficialInstitution("NH농협은행", ("NH농협은행", "농협은행", "농협"), ("nonghyup.com",)),
    OfficialInstitution("IBK기업은행", ("IBK기업은행", "기업은행"), ("ibk.co.kr",)),
    OfficialInstitution("카카오뱅크", ("카카오뱅크",), ("kakaobank.com",)),
    OfficialInstitution("토스뱅크", ("토스뱅크",), ("tossbank.com",)),
    OfficialInstitution("케이뱅크", ("케이뱅크",), ("kbanknow.com",)),
    OfficialInstitution("새마을금고", ("새마을금고",), ("kfcc.co.kr",)),
    OfficialInstitution("신협", ("신협",), ("cu.co.kr",)),
    OfficialInstitution("우체국", ("우체국", "우정사업본부"), ("epost.go.kr", "koreapost.go.kr")),
    # 카드사
    OfficialInstitution("신한카드", ("신한카드",), ("shinhancard.com",)),
    OfficialInstitution("삼성카드", ("삼성카드",), ("samsungcard.com",)),
    OfficialInstitution("현대카드", ("현대카드",), ("hyundaicard.com",)),
    OfficialInstitution("KB국민카드", ("KB국민카드", "국민카드"), ("kbcard.com",)),
    OfficialInstitution("롯데카드", ("롯데카드",), ("lottecard.co.kr",)),
    OfficialInstitution("우리카드", ("우리카드",), ("uricard.com",)),
    # 공공/사법기관
    OfficialInstitution("금융감독원", ("금융감독원",), ("fss.or.kr",)),
    OfficialInstitution("금융위원회", ("금융위원회",), ("fsc.go.kr",)),
    OfficialInstitution("경찰청", ("경찰청",), ("police.go.kr",)),
    OfficialInstitution("검찰청", ("검찰청",), ("spo.go.kr",)),
    OfficialInstitution("국세청", ("국세청",), ("nts.go.kr",)),
    OfficialInstitution("관세청", ("관세청",), ("customs.go.kr",)),
    OfficialInstitution(
        "국민건강보험공단",
        ("국민건강보험공단", "건강보험공단", "국민건강보험"),
        ("nhis.or.kr",),
    ),
)
