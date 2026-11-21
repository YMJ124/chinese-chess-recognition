from __future__ import annotations

import csv
import json
import posixpath
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .labels import PIECE_CLASSES


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _crop_from_word(image: Image.Image, src_rect: ET.Element | None) -> Image.Image:
    if src_rect is None:
        return image.copy()
    values = {name: int(src_rect.attrib.get(name, "0")) / 100000.0 for name in ("l", "t", "r", "b")}
    width, height = image.size
    box = (
        round(width * values["l"]),
        round(height * values["t"]),
        round(width * (1.0 - values["r"])),
        round(height * (1.0 - values["b"])),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"DOCX 包含无效的 Word 裁剪参数: {box}")
    return image.crop(box)


def read_docx_pictures(docx_path: str | Path) -> list[dict]:
    """按文档顺序读取 DOCX 中的图片对象，并应用 Word srcRect 裁剪。"""
    docx_path = Path(docx_path)
    with ZipFile(docx_path) as archive:
        document_root = ET.fromstring(archive.read("word/document.xml"))
        rel_root = ET.fromstring(archive.read("word/_rels/document.xml.rels"))
        relationships = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall("pr:Relationship", NS)
            if "Id" in rel.attrib and "Target" in rel.attrib
        }

        pictures: list[dict] = []
        for picture in document_root.findall(".//pic:pic", NS):
            blip = picture.find(".//a:blip", NS)
            if blip is None:
                continue
            relationship_id = blip.attrib.get(f"{{{NS['r']}}}embed")
            if not relationship_id or relationship_id not in relationships:
                continue
            target = relationships[relationship_id]
            archive_path = posixpath.normpath(posixpath.join("word", target))
            with Image.open(BytesIO(archive.read(archive_path))) as source:
                rgb = source.convert("RGB")
            cropped = _crop_from_word(rgb, picture.find(".//a:srcRect", NS))
            pictures.append(
                {
                    "relationship_id": relationship_id,
                    "source_path": archive_path,
                    "source_size": list(rgb.size),
                    "image": cropped,
                }
            )
    return pictures


def _write_reference_sheet(records: list[dict], output_path: Path) -> None:
    columns, cell = 3, 300
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
    draw = ImageDraw.Draw(sheet)
    font = _font(21)
    for index, record in enumerate(records):
        row, column = divmod(index, columns)
        image = ImageOps.contain(record["image"], (230, 230), Image.Resampling.LANCZOS)
        x = column * cell + (cell - image.width) // 2
        y = row * cell + 10
        sheet.paste(image, (x, y))
        label = PIECE_CLASSES[index].display
        bbox = draw.textbbox((0, 0), label, font=font)
        draw.text((column * cell + (cell - (bbox[2] - bbox[0])) // 2, row * cell + 250), label, fill="black", font=font)
    sheet.save(output_path, quality=95)


def extract_dataset(docx_path: str | Path, output_dir: str | Path) -> list[dict]:
    output_dir = Path(output_dir)
    base_dir = output_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    pictures = read_docx_pictures(docx_path)
    if len(pictures) != len(PIECE_CLASSES):
        raise ValueError(f"期望 DOCX 包含 14 个棋子图片对象，实际为 {len(pictures)} 个")

    manifest: list[dict] = []
    records: list[dict] = []
    for piece, picture in zip(PIECE_CLASSES, pictures, strict=True):
        filename = f"{piece.index:02d}_{piece.key}.png"
        destination = base_dir / filename
        picture["image"].save(destination)
        item = piece.to_dict() | {
            "file": str(Path("base") / filename),
            "width": picture["image"].width,
            "height": picture["image"].height,
            "source_path": picture["source_path"],
            "source_size": picture["source_size"],
        }
        manifest.append(item)
        records.append(picture)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    _write_reference_sheet(records, output_dir / "reference_sheet.jpg")
    return manifest
