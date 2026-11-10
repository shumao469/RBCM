"""Create a reproducible cross-domain target manifest before scoring."""
from pathlib import Path
import hashlib
import os
import re
import tarfile

import cv2
import numpy as np
from PIL import Image


def sha(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
        return digest.hexdigest()


def save_member(archive, member, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    content = archive.extractfile(member).read()
    if not output.exists() or output.read_bytes() != content:
        temporary = output.with_suffix(output.suffix + '.part')
        temporary.write_bytes(content)
        os.replace(temporary, output)


def signature(path):
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        gray = rgb.convert('L')
        small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        edges = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    coefficients = cv2.dct(small)[:8, :8].flatten()[1:]
    phash = int.from_bytes(np.packbits(coefficients > np.median(coefficients)).tobytes(), 'big')
    dhash = int.from_bytes(np.packbits(edges[:, 1:] > edges[:, :-1]).tobytes(), 'big')
    return digest, phash, dhash


def prior_images(official, round7):
    extensions = {'.jpg', '.jpeg', '.png', '.pgm', '.bmp'}
    folders = [official / source / 'image' for source in ('BSDS500', 'UDED', 'BIPED', 'NYUDv2', 'Multicue')]
    folders.append(round7 / 'data/BSDS500_official/images/val')
    for folder in folders:
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for file in sorted(folder.iterdir()):
            if file.is_file() and file.suffix.lower() in extensions:
                yield file


def berkeley(archive_path, output):
    expected = '97e49d31764f3912f0c4122707d53062ac9e783ba0f095e447a4d53c1a41af8e'
    if sha(archive_path) != expected:
        raise ValueError('Berkeley release archive SHA differs from the verified Round 7 archive')
    members = {'images': {}, 'groundTruth': {}}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            pattern = r'(?:^|/)BSDS500/data/(images|groundTruth)/train/(\d+)\.(jpg|mat)$'
            matched = re.search(pattern, member.name)
            if member.isfile() and matched:
                kind, sid, extension = matched.groups()
                file = output / kind / f'{sid}.{extension}'
                save_member(archive, member, file)
                members[kind][sid] = file
    if len(members['images']) != 200 or members['images'].keys() != members['groundTruth'].keys():
        raise ValueError('BSDS official train archive did not yield 200 paired images and GTs')
    return [dict(sample_id=sid, image=str(file), gt=str(members['groundTruth'][sid]))
            for sid, file in sorted(members['images'].items())]


def groningen(archive_path, output):
    """Find images and line-drawing labels by the archive's directory names."""
    if not archive_path.exists():
        return []
    images, labels = {}, {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            if not member.isfile():
                continue
            path = Path(member.name)
            if path.suffix.lower() not in {'.pgm', '.pbm', '.ppm', '.png', '.jpg'}:
                continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', path.stem):
                continue
            if len(path.parts) < 2 or path.parts[-1] != path.name:
                continue
            is_label = path.parts[-2].lower() == 'gt' and path.stem.endswith('_gt_binary')
            if not is_label and path.parts[-2].lower() != 'images':
                continue
            key = path.stem.lower().removesuffix('_gt_binary') if is_label else path.stem.lower()
            dest = output / ('labels' if is_label else 'images') / path.name
            if key in (labels if is_label else images):
                raise ValueError(f'Duplicate CID member key {key}: {member.name}')
            save_member(archive, member, dest)
            (labels if is_label else images)[key] = dest
    if len(images) != 40 or images.keys() != labels.keys():
        raise ValueError(f'Unrecognized CID structure: {len(images)} images, {len(labels)} labels; keys={list(images)[:4]}/{list(labels)[:4]}')
    return [dict(sample_id=sid, image=str(file), gt=str(labels[sid])) for sid, file in sorted(images.items())]


def adjudicate(groups, references):
    previous = [(path, signature(path)) for path in references]
    out = {}
    for dataset, entries in groups.items():
        admitted, quarantined = [], []
        for entry in entries:
            image = Path(entry['image'])
            digest, phash, dhash = signature(image)
            duplicate = []
            for path, (ref_digest, ref_phash, ref_dhash) in previous:
                ph, dh = (phash ^ ref_phash).bit_count(), (dhash ^ ref_dhash).bit_count()
                if digest == ref_digest or (ph <= 6 and dh <= 8):
                    duplicate.append(dict(path=str(path), exact=digest == ref_digest, phash_distance=ph, dhash_distance=dh))
            saved = dict(entry, image_sha256=sha(image), gt_sha256=sha(entry['gt']), decoded_sha256=digest)
            if duplicate:
                quarantined.append(dict(saved, matches=duplicate))
            else:
                admitted.append(saved)
        out[dataset] = dict(entries=admitted, quarantined=quarantined, supplied=len(entries))
    return dict(datasets=out, n_reference_images=len(previous), match_rule='decoded RGB SHA or both grayscale DCT pHash <=6 and horizontal dHash <=8; sensitive to major crops/rotations', exclusions='Compared top-level original-looking image files in staged official BSDS500, UDED, BIPED, NYUDv2, Multicue and official BSDS validation images. Earlier checkpoint exact training exposure cannot be fully reconstructed.')
