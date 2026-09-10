"""Exercise archive migration only inside disposable, isolated workspaces.

No test prepares, stages, or changes a real experiment archive. The transaction
tests inject failures around individual file moves to simulate interruption.
"""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest


MIGRATION_SOURCE = Path(__file__).resolve().parents[1] / "docs/cleanup/migrate_archives.py"


@pytest.fixture
def migration(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_archive_migration", MIGRATION_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "WORK", root / "docs/cleanup/archive_migration_2026-09-09")
    monkeypatch.setattr(module, "STAGE", root / "deletable/archive_migration_2026-09-09")
    monkeypatch.setattr(module, "PLAN", module.WORK / "plan.json.gz")
    # checked's default argument captured the real ROOT when the module loaded.
    monkeypatch.setattr(module.checked, "__defaults__", (root,))
    for kind in ("absolute", "cqr"):
        (root / "envelope_method/results" / kind).mkdir(parents=True)
    policy_path = root / "docs/cleanup/retention_policy.json"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps({
        "version": 1,
        "default_mode": "scores",
        "drop_members": sorted(module.RAW_OBSERVATION_KEYS),
        "results_root": "envelope_method/results",
        "scope": ["absolute", "cqr"],
        "keep_full": [{"kind": "absolute", "config_id": "reviewed", "trials": [0]}],
    }), encoding="utf-8")
    return module


def create_trial(module, trial=1, config="reviewed"):
    folder = module.ROOT / "envelope_method/results/absolute" / config
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f"trial_{trial:03d}.npz"
    arrays = {
        f"{variable}_{split}": np.arange(size * 2, dtype=np.float64).reshape(size, 2) + trial
        for split, size in (("train", 8), ("cal", 5), ("test", 7))
        for variable in ("X", "y")
    }
    arrays.update(
        scores_cal=np.arange(10, dtype=np.float64).reshape(5, 2) / 3,
        scores_test=np.arange(14, dtype=np.float64).reshape(7, 2) / 5,
        Envelope=np.array([2.0, np.inf]),
        model_coef=np.array([[1.0, -2.0], [0.0, 1.0]]),
        base_lengths_test=np.ones((7, 2)),
        extra_nonstandard_member=np.arange(11, dtype=np.uint8),
    )
    np.savez_compressed(archive, **arrays)
    checkpoint = dict(
        version=2,
        records=[dict(method="Envelope", trial=trial, outcome_volume=float("inf"),
                      test_coverage=6 / 7, runtime=.0123, coordinate_lengths=[4., float("inf")])],
        archive_sha256=module.sha256_file(archive),
        custom_provenance={"must_survive": "yes"},
    )
    archive.with_suffix(".json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    (folder / "trials.csv").write_text("trial,method,test_coverage\n1,Envelope,0.857142857\n")
    return archive


def prepared_trial(module):
    source = create_trial(module)
    source_bytes = source.read_bytes()
    checkpoint_bytes = source.with_suffix(".json").read_bytes()
    module.prepare()
    row, = module.read_plan()["records"]
    return source, row, source_bytes, checkpoint_bytes


def assert_complete(module, source, row, original_bytes, checkpoint_bytes):
    backup = module.STAGE / row["path"]
    assert backup.read_bytes() == original_bytes
    assert backup.with_suffix(".json").read_bytes() == checkpoint_bytes
    current = json.loads(source.with_suffix(".json").read_text())
    previous = json.loads(checkpoint_bytes)
    assert current["records"] == previous["records"]
    assert current["custom_provenance"] == previous["custom_provenance"]
    assert module.get_storage_mode(current) == "scores"
    with zipfile.ZipFile(backup) as old, zipfile.ZipFile(source) as new:
        assert set(new.namelist()) == set(row["retained_members"])
        removed = set(old.namelist()) - set(new.namelist())
        assert {name.removesuffix(".npy") for name in removed} == module.RAW_OBSERVATION_KEYS
        metadata = current["storage"]["removed_arrays"]
        assert set(metadata) == module.RAW_OBSERVATION_KEYS
        with np.load(backup, allow_pickle=False) as original_arrays:
            for name in removed:
                key = name.removesuffix(".npy")
                assert metadata[key]["npy_sha256"] == hashlib.sha256(old.read(name)).hexdigest()
                assert metadata[key]["npy_bytes"] == len(old.read(name))
                assert metadata[key]["shape"] == list(original_arrays[key].shape)
                assert metadata[key]["dtype"] == str(original_arrays[key].dtype)
        for name in new.namelist():
            assert new.read(name) == old.read(name), name
        assert not ({name.removesuffix(".npy") for name in new.namelist()} & module.RAW_OBSERVATION_KEYS)
    module.validate_archive(source, current)


def test_stage_preserves_every_retained_payload_record_and_full_exception(migration):
    preserved = create_trial(migration, trial=0)
    preserved_bytes = preserved.read_bytes()
    preserved_json = preserved.with_suffix(".json").read_bytes()
    source, row, original_bytes, checkpoint_bytes = prepared_trial(migration)
    migration.stage(workers=2)
    assert_complete(migration, source, row, original_bytes, checkpoint_bytes)
    assert preserved.read_bytes() == preserved_bytes
    assert preserved.with_suffix(".json").read_bytes() == preserved_json
    migration.verify()


def test_completed_staging_is_idempotent(migration):
    source, row, original_bytes, checkpoint_bytes = prepared_trial(migration)
    first = migration.stage_one(row)
    snapshots = {p: p.read_bytes() for p in [source, source.with_suffix(".json"),
                 migration.STAGE / row["path"], (migration.STAGE / row["path"]).with_suffix(".json")]}
    assert migration.stage_one(row) == first
    assert all(path.read_bytes() == value for path, value in snapshots.items())
    assert_complete(migration, source, row, original_bytes, checkpoint_bytes)


@pytest.mark.parametrize("damage", ["missing_archive", "corrupt_archive", "changed_checkpoint"])
def test_changed_sources_fail_before_any_move(migration, damage):
    source, row, _, _ = prepared_trial(migration)
    if damage == "missing_archive":
        source.unlink()
    elif damage == "corrupt_archive":
        with source.open("ab") as stream:
            stream.write(b"changed after prepare")
    else:
        source.with_suffix(".json").write_text("{}")
    expected_active = source.read_bytes() if source.exists() else None
    expected_checkpoint = source.with_suffix(".json").read_bytes()
    with pytest.raises((ValueError, FileNotFoundError)):
        migration.stage_one(row)
    assert not (migration.STAGE / row["path"]).exists()
    assert not (migration.STAGE / row["path"]).with_suffix(".json").exists()
    assert (source.read_bytes() if source.exists() else None) == expected_active
    assert source.with_suffix(".json").read_bytes() == expected_checkpoint


def test_prepare_refuses_corrupted_source_checkpoint_pair(migration):
    source = create_trial(migration)
    with source.open("ab") as stream:
        stream.write(b"hash mismatch")
    with pytest.raises(ValueError, match="hash differs"):
        migration.prepare()
    assert source.exists()
    assert source.with_suffix(".json").exists()
    assert not migration.PLAN.exists()


def test_prepare_refuses_unreviewed_nonempty_configuration(migration):
    source = create_trial(migration, config="unreviewed")
    with pytest.raises(ValueError, match="Unreviewed"):
        migration.prepare()
    assert source.exists()
    assert not migration.PLAN.exists()


@pytest.mark.parametrize("relative", ["../escape.npz", ".git/config", ".codex/settings"])
def test_checked_rejects_escape_and_protected_paths(migration, relative):
    with pytest.raises(ValueError):
        migration.checked(migration.ROOT / relative)


def test_all_plan_paths_are_checked_before_first_move(migration):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    plan = migration.read_plan()
    plan["records"].append({**row, "path": "../outside.npz"})
    with gzip.open(migration.PLAN, "wt", encoding="utf-8") as stream:
        json.dump(plan, stream)
    with pytest.raises(ValueError, match="outside"):
        migration.stage(workers=2)
    assert source.read_bytes() == source_bytes
    assert source.with_suffix(".json").read_bytes() == checkpoint_bytes
    assert not (migration.STAGE / row["path"]).exists()


@pytest.mark.parametrize("interrupt_after_move", [1, 2, 3])
def test_interruption_after_each_move_resumes(migration, monkeypatch, interrupt_after_move):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    real_rename = Path.rename
    count = 0

    def interrupted_rename(path, target):
        nonlocal count
        result = real_rename(path, target)
        count += 1
        if count == interrupt_after_move:
            raise RuntimeError("simulated interruption after move")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "rename", interrupted_rename)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            migration.stage_one(row)
    migration.stage_one(row)
    assert_complete(migration, source, row, source_bytes, checkpoint_bytes)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_bad_pending_replacement_does_not_orphan_intact_active_source(migration, monkeypatch, damage):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    original_atomic = migration.atomic_json

    def stop_after_transaction(path, data):
        original_atomic(path, data)
        if str(path).endswith(".transaction.json") and data.get("status") == "prepared":
            raise RuntimeError("simulated stop before first move")

    with monkeypatch.context() as scoped:
        scoped.setattr(migration, "atomic_json", stop_after_transaction)
        with pytest.raises(RuntimeError, match="before first move"):
            migration.stage_one(row)
    pending = source.with_suffix(".scores.pending.npz")
    if damage == "missing":
        pending.unlink()
    else:
        pending.write_bytes(b"corrupted pending replacement")
    try:
        migration.stage_one(row)
    except (ValueError, FileNotFoundError):
        # A safe preflight rejection is acceptable; moving both intact inputs
        # away before discovering the bad replacement is not.
        assert source.is_file(), "Intact active archive was orphaned by failed recovery"
        assert source.read_bytes() == source_bytes
        assert source.with_suffix(".json").read_bytes() == checkpoint_bytes
    else:
        # Rebuilding from the intact original is also acceptable.
        assert_complete(migration, source, row, source_bytes, checkpoint_bytes)


def test_verifier_compares_payload_to_original_not_only_active_metadata(migration):
    source, row, _, _ = prepared_trial(migration)
    migration.stage_one(row)
    with zipfile.ZipFile(source) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    # Change an actual numerical byte, then make active metadata internally
    # self-consistent. The staged original remains the independent evidence.
    changed = bytearray(members["Envelope.npy"])
    changed[-1] ^= 1
    members["Envelope.npy"] = bytes(changed)
    with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    checkpoint_path = source.with_suffix(".json")
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["archive_sha256"] = migration.sha256_file(source)
    checkpoint["storage"]["archive_sha256"] = checkpoint["archive_sha256"]
    checkpoint["storage"]["archive_bytes"] = source.stat().st_size
    with zipfile.ZipFile(source) as archive:
        checkpoint["storage"]["retained_payload_sha256"] = migration.payload_digest(archive, row["retained_members"])
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError):
        migration.verify()


def test_pending_replacement_can_be_rebuilt_after_original_was_moved(migration, monkeypatch):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    real_rename = Path.rename

    def stop_after_original(path, target):
        result = real_rename(path, target)
        if path == source:
            raise RuntimeError("interrupted after staging original")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "rename", stop_after_original)
        with pytest.raises(RuntimeError, match="staging original"):
            migration.stage_one(row)
    assert not source.exists()
    source.with_suffix(".scores.pending.npz").unlink()
    migration.stage_one(row)
    assert_complete(migration, source, row, source_bytes, checkpoint_bytes)


def test_restore_round_trip_and_idempotence_keep_both_versions(migration):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    migration.stage_one(row)
    scores_bytes = source.read_bytes()
    scores_checkpoint = source.with_suffix(".json").read_bytes()
    migration.restore()
    reduced = migration.STAGE / "replaced_scores" / row["path"]
    assert source.read_bytes() == source_bytes
    assert source.with_suffix(".json").read_bytes() == checkpoint_bytes
    assert reduced.read_bytes() == scores_bytes
    assert reduced.with_suffix(".json").read_bytes() == scores_checkpoint
    migration.restore()
    assert source.read_bytes() == source_bytes
    assert source.with_suffix(".json").read_bytes() == checkpoint_bytes
    assert reduced.read_bytes() == scores_bytes
    with pytest.raises(ValueError, match="[Rr]estored|restore"):
        migration.stage_one(row)
    assert source.read_bytes() == source_bytes
    assert source.with_suffix(".json").read_bytes() == checkpoint_bytes


@pytest.mark.parametrize("interrupt_after_move", [1, 2, 3, 4])
def test_restore_interruption_after_each_move_resumes(migration, monkeypatch, interrupt_after_move):
    source, row, source_bytes, checkpoint_bytes = prepared_trial(migration)
    migration.stage_one(row)
    reduced_bytes = source.read_bytes()
    reduced_checkpoint_bytes = source.with_suffix(".json").read_bytes()
    real_rename = Path.rename
    count = 0

    def interrupted_rename(path, target):
        nonlocal count
        result = real_rename(path, target)
        count += 1
        if count == interrupt_after_move:
            raise RuntimeError("simulated restore interruption")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "rename", interrupted_rename)
        with pytest.raises(RuntimeError, match="restore interruption"):
            migration.restore()
    migration.restore()
    reduced = migration.STAGE / "replaced_scores" / row["path"]
    assert source.read_bytes() == source_bytes
    assert source.with_suffix(".json").read_bytes() == checkpoint_bytes
    assert reduced.read_bytes() == reduced_bytes
    assert reduced.with_suffix(".json").read_bytes() == reduced_checkpoint_bytes


@pytest.mark.parametrize("target", ["backup_npz", "backup_json", "active_npz", "active_json"])
def test_restore_rejects_corruption_before_moving_any_trial(migration, target):
    first = create_trial(migration, trial=1)
    second = create_trial(migration, trial=2)
    migration.prepare()
    rows = migration.read_plan()["records"]
    for row in rows:
        migration.stage_one(row)
    row = next(row for row in rows if row["path"].endswith("trial_002.npz"))
    backup = migration.STAGE / row["path"]
    damaged = {
        "backup_npz": backup,
        "backup_json": backup.with_suffix(".json"),
        "active_npz": second,
        "active_json": second.with_suffix(".json"),
    }[target]
    with damaged.open("ab") as stream:
        stream.write(b"post-stage change")
    observed = {path: path.read_bytes() for path in [first, first.with_suffix(".json"),
                second, second.with_suffix(".json"), damaged]}
    with pytest.raises(ValueError):
        migration.restore()
    assert all(path.read_bytes() == expected for path, expected in observed.items())
    assert not (migration.STAGE / "replaced_scores").exists()


def test_restore_preflights_json_paths_through_the_same_path_guard(migration, monkeypatch):
    source, row, _, _ = prepared_trial(migration)
    migration.stage_one(row)
    protected_checkpoint = source.with_suffix(".json")
    source_bytes = source.read_bytes()
    checkpoint_bytes = protected_checkpoint.read_bytes()
    real_checked = migration.checked

    def guarded_path(path, *args, **kwargs):
        if Path(path) == protected_checkpoint:
            # Model the existing path guard rejecting a redirected JSON path.
            # This avoids creating OS symlinks or requiring Windows privileges.
            raise ValueError("Redirected filesystem path: active checkpoint")
        return real_checked(path, *args, **kwargs)

    monkeypatch.setattr(migration, "checked", guarded_path)
    with pytest.raises(ValueError, match="Redirected"):
        migration.restore()
    assert source.read_bytes() == source_bytes
    assert protected_checkpoint.read_bytes() == checkpoint_bytes
