"""Admit only independent-source contours after inventory finds old BSDS train assets."""
from pathlib import Path
import re

from admission import adjudicate, berkeley, groningen, sha
from run import BSDS, CACHE, CID, OLD_CACHE, OLD_STAGE, OUT, frozen


def original_references(official, train):
    yield from (Path(item['image']) for item in train)
    yield from sorted((OLD_CACHE / 'data/BSDS500_official/images/val').glob('*.jpg'))
    patterns = {
        'BSDS500': r'\d+\.jpg',
        'BIPED': r'RGB_\d+\.(jpg|png)',
        'NYUDv2': r'(?:img_\d+|train_Images_images_img_\d+_scale100)\.png',
        'Multicue': r'\d+_left_\d+\.png',
        'UDED': r'.+\.(png|jpg|jpeg)',
    }
    for source, pattern in patterns.items():
        folder = official / source / 'image'
        for image in sorted(folder.iterdir()):
            if re.fullmatch(pattern, image.name, re.IGNORECASE):
                yield image


def main():
    official = OLD_STAGE / 'asset_staging/data_d1c9dd8694dc/RBCM-Edge-Data/edge_data/official_rbcm'
    bsds_archive = OLD_CACHE / 'downloads/BSR_bsds500.tgz'
    cid_archive = CACHE / 'incoming/CID_contours.tar.gz'
    train = berkeley(bsds_archive, CACHE / 'datasets/bsds_train')
    release_dir = official / 'BSDS500/image'
    known_augmented = set()
    for path in release_dir.iterdir():
        match = re.fullmatch(r'train_aug_data_.+_(\d+)\.jpg', path.name)
        if match:
            known_augmented.add(match[1])
    bsds_overlap = sorted(item['sample_id'] for item in train if item['sample_id'] in known_augmented)
    if len(bsds_overlap) < 1:
        raise ValueError('Expected to verify staged release BSDS train assets before exclusion')
    cid = groningen(cid_archive, CACHE / 'datasets/cid')
    audit = adjudicate({CID: cid}, original_references(official, train))
    audit['exclusions'] = 'Original-looking frames of prior BSDS500 val/test and released BSDS train (official originals for augmented IDs), UDED, BIPED, NYUDv2, Multicue. Near duplicate rule can miss major crops and rotations. Exact earlier backbone training exposure not fully reconstructable.'
    audit['admission_implementation_sha256'] = sha(__file__)
    audit['sources'] = {
        CID: dict(url='https://www.cs.rug.nl/~imaging/databases/contour_database/contour_database.html',
                  archive_sha256=sha(cid_archive), scope='Independent corpus with single hand-drawn contour labels; separate annotation conventions.',
                  acquisition='admitted after overlap quarantine'),
        BSDS: dict(url='https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html',
                   archive_sha256=sha(bsds_archive), scope='Official BSDS train files present as derived assets in the existing RBCM release; conservatively excluded as a fresh confirmation target.',
                   acquisition='reference images for CID duplicate screening only', n_supplied=len(train),
                   n_overlapping_release_augmented=len(bsds_overlap), overlapping_sample_ids=bsds_overlap)
    }
    path = OUT / 'target_admission.json'
    frozen(path, audit)
    result = audit['datasets'][CID]
    print(f"CID admitted {len(result['entries'])}/{result['supplied']}; quarantined {[x['sample_id'] for x in result['quarantined']]}; prior comparison references {audit['n_reference_images']}; staged release overlap BSDS train {len(bsds_overlap)}/{len(train)}", flush=True)


if __name__ == '__main__':
    main()
