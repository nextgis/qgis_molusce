import pickle
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
from typing import TYPE_CHECKING, Dict, List, Set, Union

from qgis.PyQt.QtCore import QCoreApplication
from qgis.utils import pluginMetadata

from molusce.algorithms.dataprovider import Raster
from molusce.algorithms.models.mce.mce import MCE
from molusce.algorithms.models.mlp.manager import MlpManager
from molusce.algorithms.models.woe.manager import WoeManager

if TYPE_CHECKING:
    from molusce.algorithms.models.lr.lr import LR

is_scipy_missed = False
if find_spec("scipy") is not None:
    from molusce.algorithms.models.lr.lr import LR
else:
    is_scipy_missed = True


class SerializerError(Exception):
    """
    Custom exception for serializer-related errors.

    :param msg: Error message describing the issue.
    """

    def __init__(self, msg: str):
        self.msg = msg


@dataclass
class ModelParams:
    """
    Represents the parameters of a model used in the application.

    :param model_type: Type of the model (e.g., ANN, WOE, LR, MCE).
    :param model: The model instance.
    :param base_xsize: X-dimension of the base raster.
    :param base_ysize: Y-dimension of the base raster.
    :param base_classes: Set of unique classes in the base raster.
    :param factors_metadata: Metadata for the input factors.
    :param molusce_version: Version of the MOLUSCE plugin.
    :param creation_ts: Timestamp of the model creation.
    """

    model_type: str
    model: Union[MlpManager, WoeManager, "LR", MCE]
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
        """
        Check if the model parameters are consistent with the input data.

        :param inputs_initial: Initial raster data.
        :param inputs_factors: Dictionary of input factor rasters.

        :return: True if consistent, False otherwise.
        """
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
        inputs_model: Union[MlpManager, WoeManager, "LR", MCE],
        inputs_initial: Raster,
        inputs_factors: Dict[str, Raster],
    ) -> "ModelParams":
        """
        Create a ModelParams instance from input data.

        :param inputs_model: The model instance (e.g., MlpManager, WoeManager, LR, MCE).
        :param inputs_initial: The initial raster data.
        :param inputs_factors: A dictionary of factor rasters.

        :return: A new ModelParams instance.

        :raises SerializerError: If the model type or inputs are invalid.
        """
        if isinstance(inputs_model, MlpManager):
            model_type = "Artificial Neural Network (Multi-layer Perceptron)"
        elif isinstance(inputs_model, WoeManager):
            model_type = "Weights of Evidence"
        elif not is_scipy_missed and isinstance(inputs_model, LR):
            model_type = "Logistic Regression"
        elif isinstance(inputs_model, MCE):
            model_type = "Multi Criteria Evaluation"
        else:
            raise SerializerError(
                QCoreApplication.translate("Serializer", "Model is unknown")
            )

        if not isinstance(inputs_initial, Raster):
            raise SerializerError(
                QCoreApplication.translate(
                    "Serializer", "Invalid initial raster"
                )
            )

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
                        QCoreApplication.translate(
                            "Serializer", "Invalid factors. {}"
                        ).format(error)
                    ) from error

        except Exception as error:
            raise SerializerError(
                QCoreApplication.translate(
                    "Serializer", "Invalid factors. {}"
                ).format(error)
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
    """
    Handles serialization and deserialization of ModelParams instances.
    """

    @classmethod
    def from_file(cls, file_path: str) -> ModelParams:
        """
        Load a ModelParams instance from a file.

        :param file_path: Path to the file containing the serialized ModelParams.

        :return: The deserialized ModelParams instance.

        :raises SerializerError: If the file is invalid or the model type is unsupported.
        """
        try:
            with open(file_path, "rb") as file:
                model_params: ModelParams = pickle.load(file)
        except ModuleNotFoundError as error:
            # fmt: off
            raise SerializerError(
                QCoreApplication.translate(
                    "Serializer",
                    "scipy is required to load Logistic Regression model"
                )
            ) from error
            # fmt: on
        except Exception as error:
            raise SerializerError(
                QCoreApplication.translate(
                    "Serializer", "Invalid file. {}"
                ).format(error)
            ) from error

        model_types = (MlpManager, WoeManager, MCE)
        if not is_scipy_missed:
            model_types += (LR,)

        if not isinstance(model_params.model, model_types):
            raise SerializerError(
                QCoreApplication.translate("Serializer", "Invalid model type")
            )

        return model_params

    @classmethod
    def to_file(cls, model_params: ModelParams, file_path: str) -> None:
        """
        Save a ModelParams instance to a file.

        :param model_params: The ModelParams instance to serialize.
        :param file_path: Path to the file where the ModelParams will be saved.

        :raises SerializerError: If an error occurs during the file writing process.
        """
        try:
            with open(file_path, "wb") as file:
                pickle.dump(model_params, file)
        except Exception as error:
            raise SerializerError(
                QCoreApplication.translate(
                    "Serializer", "An error occurred while writing data: {}"
                ).format(error)
            ) from error
