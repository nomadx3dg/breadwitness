
from breadwitness.hashing import (
    compute_perceptual_hashes,
    hamming_distance,
    hamming_distances,
    sha256_file,
)

from .conftest import make_image


def test_hamming_distance_basic():
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64
    assert hamming_distance("0000000000000001", "0000000000000003") == 1


def test_vectorized_matches_scalar():
    query = "f0f0f0f0f0f0f0f0"
    candidates = ["0000000000000000", "ffffffffffffffff", "f0f0f0f0f0f0f0f1", query]
    expected = [hamming_distance(query, c) for c in candidates]
    assert hamming_distances(query, candidates).tolist() == expected
    assert hamming_distances(query, []).shape == (0,)


def test_vectorized_handles_high_bit():
    assert hamming_distances("8000000000000000", ["0000000000000000"]).tolist() == [1]


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert sha256_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_perceptual_hashes_are_stable_and_16_hex_chars():
    h1 = compute_perceptual_hashes(make_image(7))
    h2 = compute_perceptual_hashes(make_image(7))
    assert h1 == h2
    for value in (h1.phash, h1.dhash, h1.ahash):
        assert len(value) == 16
        int(value, 16)
