import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Set, Union

from qgis.utils import pluginMetadata

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.lr.lr import LR
from molusce.algorithms.models.mce.mce import MCE
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.woe.manager import WoeManager


class SerializerError(Exception):
    def __init__(self, msg):
        self.msg = msg


@dataclass
class ModelParams:
    model_type: str
    model: Union[MlpManager, WoeManager, LR, MCE]
    base_xsize: int
    base_ysize: int
    base_classes: Set
    factors_metadata: List[Dict[str, int]]
    molusce_version: str
    creation_ts: datetime

    def is_consistent_with(
        self,
        inputs_initial: Raster,
        inputs_factors: Dict[str, Raster],
    ) -> bool:
        if (self.base_xsize, self.base_ysize) != (
            inputs_initial.getXSize(),
            inputs_initial.getYSize(),
        ):
            return False

        if self.base_classes != inputs_initial.getUniqueValues():
            return False

        if len(self.factors_metadata) != len(inputs_factors) or not all(
            loaded_factor["bandcount"] == input_factor.bandcount
            for loaded_factor, input_factor in zip(
                self.factors_metadata, inputs_factors.values()
            )
        ):
            return False

        return True

    @classmethod
    def from_data(
        cls,
        inputs_model: Union[MlpManager, WoeManager, LR, MCE],
        inputs_initial: Raster,
        inputs_factors: Dict[str, Raster],
    ) -> "ModelParams":
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
                except Exception as error:
                    raise SerializerError(
                        "Invalid factor. %s" % str(error)
                    ) from error

        except Exception as error:
            raise SerializerError(
                "Invalid factors. %s" % str(error)
            ) from error

        return ModelParams(
            model_type,
            inputs_model,
            inputs_initial.getXSize(),
            inputs_initial.getYSize(),
            inputs_initial.getUniqueValues(),
            factors_metadata,
            molusce_version=pluginMetadata("molusce", "version"),
            creation_ts=datetime.now(),
        )


class ModelParamsSerializer:
    @classmethod
    def from_file(cls, file_path: str) -> ModelParams:
        try:
            with open(file_path, "rb") as file:
                model_params: ModelParams = pickle.load(file)
        except Exception as error:
            raise SerializerError("Invalid file. %s" % str(error)) from error

        if not isinstance(
            model_params.model, (MlpManager, WoeManager, LR, MCE)
        ):
            raise SerializerError("Invalid model type")

        return model_params

    @classmethod
    def to_file(cls, model_params: ModelParams, file_path: str) -> None:
        try:
            with open(file_path, "wb") as file:
                pickle.dump(model_params, file)

        except Exception as error:
            raise SerializerError(
                "An error occurred while writing data"
            ) from error
