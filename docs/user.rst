User API
########

CellFlow model
~~~~~~~~~~~~~~

.. module:: cfp.model
.. currentmodule:: cfp.model
.. autosummary::
    :toctree: genapi

    CellFlow

Solvers
~~~~~~~

.. module:: cfp.solvers
.. currentmodule:: cfp.solvers
.. autosummary::
    :toctree: genapi

    OTFlowMatching
    GENOT


Networks
~~~~~~~~
.. module:: cfp.networks
.. currentmodule:: cfp.networks
.. autosummary::
    :toctree: genapi

    ConditionalVelocityField
    ConditionEncoder
    SelfAttention
    SeedAttentionPooling
    TokenAttentionPooling
    MLPBlock
    SelfAttentionBlock

Utils
~~~~~
.. module:: cfp.utils
.. currentmodule:: cfp.utils
.. autosummary::
    :toctree: genapi

    match_linear

Training
~~~~~
.. module:: cfp.training
.. currentmodule:: cfp.training
.. autosummary::
    :toctree: genapi

    BaseCallback
    CallbackRunner
    ComputationCallback
    LoggingCallback
    Metrics
    PCADecodedMetrics
    VAEDecodedMetrics
    WandbLogger
    CellFlowTrainer

Plotting
~~~~~
.. module:: cfp.plotting
.. currentmodule:: cfp.plotting
.. autosummary::
    :toctree: genapi

    plot_condition_embedding

Preprocessing
~~~~~
.. module:: cfp.preprocessing
.. currentmodule:: cfp.preprocessing
.. autosummary::
    :toctree: genapi

    centered_pca
    project_pca
    reconstruct_pca
    annotate_compounds
    encode_onehot
    get_molecular_fingerprints
    compute_wknn
    transfer_labels

External
~~~~~
.. module:: cfp.external
.. currentmodule:: cfp.external
.. autosummary::
    :toctree: genapi

    CFJaxSCVI
