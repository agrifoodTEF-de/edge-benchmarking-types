"""Contract tests for the ``BenchmarkConfig.inference_client`` discriminated union.

Two failure modes are guarded here, both of which are silent rather than loud:

1. Without a discriminator, pydantic's smart union validates a minimal BirdNET
   payload as a ``TritonDenseNetClient`` and drops every BirdNET parameter
   without raising.
2. With a *bare* discriminator, any producer that predates ``client_type``
   (notably the Agri-Gaia frontend, which builds the payload dict by hand) gets
   ``union_tag_not_found`` and every existing DenseNet/YOLO job fails with a 422.

The payloads in ``test_untagged_*`` are shaped exactly like the ones
``BenchmarkJobCreateDialog.tsx::createEdgeBenchmarkStartPayload`` produces. If
they start failing, the platform is about to break.
"""

import pytest
from pydantic import ValidationError

from edge_benchmarking_types.edge_farm.models import (
    BenchmarkConfig,
    TritonBirdNetClient,
    TritonDenseNetClient,
    TritonYoloClient,
)

EDGE_DEVICE = {"host": "edge-03"}


def build(inference_client: dict) -> BenchmarkConfig:
    return BenchmarkConfig(edge_device=EDGE_DEVICE, inference_client=inference_client)


@pytest.mark.parametrize(
    ("client_type", "extra", "expected"),
    [
        ("TritonDenseNetClient", {}, TritonDenseNetClient),
        (
            "TritonYoloClient",
            {"input_width": 640, "input_height": 640},
            TritonYoloClient,
        ),
        ("TritonBirdNetClient", {}, TritonBirdNetClient),
    ],
)
def test_tagged_payload_resolves_exactly(client_type, extra, expected):
    config = build({"host": "edge-03", "client_type": client_type, **extra})
    assert isinstance(config.inference_client, expected)


def test_tagged_birdnet_needs_no_distinguishing_fields():
    """The case a smart union cannot get right.

    A BirdNET payload carrying only defaults is indistinguishable from a
    DenseNet one by shape alone -- the tag is what disambiguates it.
    """
    config = build({"host": "edge-03", "client_type": "TritonBirdNetClient"})
    assert isinstance(config.inference_client, TritonBirdNetClient)


def test_untagged_densenet_still_accepted():
    """Back-compat: the payload Agri-Gaia sends for a DenseNet job today."""
    config = build(
        {
            "protocol": "http",
            "host": "edge-03",
            "port": 8000,
            "num_workers": 1,
            "samples_per_second": None,
            "batch_size": 1,
            "warm_up": False,
            "num_classes": 1,
            "scaling": "inception",
        }
    )
    assert isinstance(config.inference_client, TritonDenseNetClient)
    assert config.inference_client.scaling == "inception"


def test_untagged_yolo_still_accepted():
    """Back-compat: the payload Agri-Gaia sends for a YOLO job today."""
    config = build(
        {
            "protocol": "http",
            "host": "edge-03",
            "port": 8000,
            "num_workers": 1,
            "samples_per_second": None,
            "batch_size": 1,
            "warm_up": False,
            "num_classes": 1,
            "scaling": None,
            "confidence_thres": 0.25,
            "iou_thres": 0.45,
            "input_width": 640,
            "input_height": 640,
        }
    )
    assert isinstance(config.inference_client, TritonYoloClient)
    assert config.inference_client.input_width == 640


def test_untagged_minimal_defaults_to_densenet():
    """An untagged payload with nothing distinguishing keeps the historical type."""
    assert isinstance(build({"host": "edge-03"}).inference_client, TritonDenseNetClient)


@pytest.mark.parametrize(
    "hint", ["sample_rate", "segment_seconds", "top_k", "overlap_seconds"]
)
def test_untagged_birdnet_hints_resolve_to_birdnet(hint):
    defaults = {
        "sample_rate": 32_000,
        "segment_seconds": 3.0,
        "top_k": 5,
        "overlap_seconds": 0.0,
    }
    config = build({"host": "edge-03", hint: defaults[hint]})
    assert isinstance(config.inference_client, TritonBirdNetClient)


def test_round_trip_preserves_birdnet_parameters():
    """The regression the discriminator exists to prevent.

    Under a smart union this silently produced a DenseNet client with
    ``top_k``/``confidence_thres`` dropped on the floor.
    """
    original = TritonBirdNetClient(host="edge-03", top_k=7, confidence_thres=0.25)
    restored = build(original.model_dump(mode="json")).inference_client

    assert isinstance(restored, TritonBirdNetClient)
    assert restored.top_k == 7
    assert restored.confidence_thres == 0.25


def test_model_dump_carries_the_tag():
    """Guards the edge-farm side: ``_create_edge_clients`` splats this dump into
    the runtime client constructor, so it must exclude ``client_type`` or every
    client raises ``TypeError`` on an unexpected kwarg."""
    assert "client_type" in TritonBirdNetClient(host="edge-03").model_dump()


def test_validator_does_not_mutate_caller_payload():
    payload = {"host": "edge-03", "sample_rate": 32_000}
    build(payload)
    assert "client_type" not in payload


@pytest.mark.parametrize(
    "kwargs",
    [
        {"overlap_seconds": 3.0},  # equal to segment_seconds
        {"overlap_seconds": 4.0},  # longer than segment_seconds
        {"bandpass_fmin": 8_000, "bandpass_fmax": 1_000},
        {"top_k": 0},
        {"confidence_thres": 1.5},
    ],
)
def test_birdnet_rejects_incoherent_configuration(kwargs):
    with pytest.raises(ValidationError):
        TritonBirdNetClient(host="edge-03", **kwargs)
