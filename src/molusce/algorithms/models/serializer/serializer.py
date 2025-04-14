import pickle
from datetime import datetime

from qgis.utils import pluginMetadata

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.lr.lr import LR
from molusce.algorithms.models.mce.mce import MCE
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.woe.manager import WoeManager


class SerializerError(Exception):
    def __init__(self, msg):
        self.msg = msg


class Serializer:
    model = None
    base_xsize = None
    base_ysize = None
    base_classes = None
    factors_metadata = None
    model_type = None
    creation_ts = None
    molusce_version = None

    def __init__(
        self,
        model,
        base_xsize,
        base_ysize,
        base_classes,
        factors_metadata,
        model_type,
        creation_ts,
        molusce_version,
    ):
        self.model = model
        self.base_xsize = base_xsize
        self.base_ysize = base_ysize
        self.base_classes = base_classes
        self.factors_metadata = factors_metadata
        self.model_type = model_type
        self.creation_ts = creation_ts
        self.molusce_version = molusce_version

    @classmethod
    def from_file(cls, file_path: str):
        try:
            with open(file_path, "rb") as f:
                imported_model = pickle.load(f)
        except Exception as e:
            raise SerializerError("Invalid file. %s" % str(e)) from e

        if not all(
            k in imported_model
            for k in (
                "model",
                "base_xsize",
                "base_ysize",
                "base_classes",
                "factors_metadata",
                "model_type",
                "creation_ts",
                "molusce_version",
            )
        ):
            raise SerializerError("Invalid file")

        if (
            not isinstance(imported_model["model"], MlpManager)
            and not isinstance(imported_model["model"], WoeManager)
            and not isinstance(imported_model["model"], LR)
            and not isinstance(imported_model["model"], MCE)
        ):
            raise SerializerError("Invalid model type")

        return cls(
            imported_model["model"],
            imported_model["base_xsize"],
            imported_model["base_ysize"],
            imported_model["base_classes"],
            imported_model["factors_metadata"],
            imported_model["model_type"],
            imported_model["creation_ts"],
            imported_model["molusce_version"],
        )

    @classmethod
    def from_data(cls, inputs_model, inputs_initial, inputs_factors):
        if isinstance(inputs_model, MlpManager):
            model_type = "Artificial Neural Network (Multi-layer Perceptron)"
        elif isinstance(inputs_model, WoeManager):
            model_type = "Weights of Evidence"
        elif isinstance(inputs_model, LR):
            model_type = "Logistic Regression"
        elif isinstance(inputs_model, MCE):
            model_type = "Multi Criteria Evaluation"
        else:
            raise SerializerError("Model is unknown")

        if not isinstance(inputs_initial, Raster):
            raise SerializerError("Invalid initial raster")

        factors_metadata = []
        try:
            for factor_name, factor_content in inputs_factors.items():
                try:
                    factors_metadata.append(
                        {
                            "name": factor_name,
                            "bandcount": factor_content.bandcount,
                        }
                    )
                except Exception as e:
                    raise SerializerError("Invalid factor. %s" % str(e)) from e
        except Exception as e:
            raise SerializerError("Invalid factors. %s" % str(e)) from e

        creation_ts = datetime.now()
        molusce_version = pluginMetadata("molusce", "version")

        return cls(
            inputs_model,
            inputs_initial.getXSize(),
            inputs_initial.getYSize(),
            inputs_initial.getUniqueValues(),
            factors_metadata,
            model_type,
            creation_ts,
            molusce_version,
        )

    def to_file(self, file_path: str) -> None:
        with open(file_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "base_xsize": self.base_xsize,
                    "base_ysize": self.base_ysize,
                    "base_classes": self.base_classes,
                    "factors_metadata": self.factors_metadata,
                    "model_type": self.model_type,
                    "creation_ts": self.creation_ts,
                    "molusce_version": self.molusce_version,
                },
                f,
            )
