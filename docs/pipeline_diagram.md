# Pipeline architecture

The pipeline keeps raw time series, quality-control evidence, trial/phase features, and modeling
artifacts separate. Explicit configuration and manifest records connect the stages.

```mermaid
flowchart LR
    A["YAML configuration"] --> C["Validated trial manifest"]
    B["Authorized modality files"] --> C
    C --> D["Modality loaders and QC"]
    D --> E["Explicit reference-clock alignment"]
    E --> F["Manual TUG phase validation"]
    F --> G["Trial- and phase-level features"]
    G --> H["Outer feature fusion"]
    H --> I["Participant-grouped baselines"]
    D --> J["Aggregate research report"]
    E --> J
    G --> J
    H --> J
    I --> J

    K["Private raw and derived data"] -. "ignored by Git" .-> B
    L["Synthetic public fixtures"] --> B
```

The implementation never assumes that modality clocks begin together. The configured mapping is
`reference_time = native_time + offset_seconds`, and every applied offset is recorded with its
method, uncertainty, operator, and notes. Modeling transformations are fitted inside each
training fold, and trials are grouped by participant.

This diagram describes software flow, not clinical validity. The public demonstration contains
no video and no identifiable participant recording.
