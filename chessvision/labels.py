from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PieceClass:
    index: int
    key: str
    faction: str
    name: str
    score: int

    @property
    def display(self) -> str:
        return f"{self.faction}-{self.name}-{self.score}分"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["display"] = self.display
        return data


# 顺序必须与象棋样本.docx 中的 14 个 Word 图片对象一致。
PIECE_CLASSES = (
    PieceClass(0, "black_jiang", "黑", "将", 10),
    PieceClass(1, "black_shi", "黑", "士", 3),
    PieceClass(2, "black_xiang", "黑", "象", 4),
    PieceClass(3, "black_ma", "黑", "馬", 7),
    PieceClass(4, "black_pao", "黑", "炮", 6),
    PieceClass(5, "black_che", "黑", "車", 9),
    PieceClass(6, "black_zu", "黑", "卒", 1),
    PieceClass(7, "red_shuai", "红", "帅", 10),
    PieceClass(8, "red_shi", "红", "仕", 3),
    PieceClass(9, "red_xiang", "红", "相", 4),
    PieceClass(10, "red_ma", "红", "馬", 7),
    PieceClass(11, "red_pao", "红", "炮", 6),
    PieceClass(12, "red_che", "红", "車", 9),
    PieceClass(13, "red_bing", "红", "兵", 1),
)

CLASS_BY_INDEX = {item.index: item for item in PIECE_CLASSES}
CLASS_BY_KEY = {item.key: item for item in PIECE_CLASSES}
