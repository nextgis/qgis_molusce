from pathlib import Path
from typing import Dict, Optional

from qgis.core import QgsSettings
from qgis.PyQt import uic
from qgis.PyQt.QtCore import QLocale
from qgis.PyQt.QtWidgets import QDialog, QWidget
from qgis.utils import pluginMetadata

FORM_CLASS, _ = uic.loadUiType(
    str(Path(__file__).parent / "ui/aboutdialogbase.ui")
)


class AboutDialog(QDialog, FORM_CLASS):
    def __init__(self, package_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setupUi(self)
        self.__package_name = package_name

        metadata = self.__metadata()

        self.pluginName.setText(self.pluginName.text().format_map(metadata))
        self.setWindowTitle(self.windowTitle().format_map(metadata))
        self.textBrowser.setHtml(self.__html(metadata))

    def __locale(self) -> str:
        override_locale = QgsSettings().value(
            "locale/overrideFlag", defaultValue=False, type=bool
        )
        if not override_locale:
            locale_full_name = QLocale.system().name()
        else:
            locale_full_name = QgsSettings().value("locale/userLocale", "")

        return locale_full_name[0:2]

    def __metadata(self) -> Dict[str, Optional[str]]:
        locale = self.__locale()
        is_ru = locale in ["ru", "uk"]

        def metadata_value(key: str) -> Optional[str]:
            value = pluginMetadata(self.__package_name, f"{key}[{locale}]")
            if value == "__error__":
                value = pluginMetadata(self.__package_name, key)
            if value == "__error__":
                value = None
            return value

        about = metadata_value("about")
        assert about is not None
        for about_stop_phrase in (
            "Разработан",
            "Developed by",
            "Développé par",
            "Desarrollado por",
            "Sviluppato da",
            "Asia Air Surveyと",
            "Desenvolvido por",
        ):
            if about.find(about_stop_phrase) > 0:
                about = about[: about.find(about_stop_phrase)]

        url = f"https://nextgis.{'ru' if is_ru else 'com'}"

        return {
            "plugin_name": metadata_value("name"),
            "description": metadata_value("description"),
            "about": about,
            "authors": metadata_value("author"),
            "video_url": metadata_value("video"),
            "homepage_url": metadata_value("repository"),
            "tracker_url": metadata_value("tracker"),
            "main_url": url,
            "data_url": url.replace("://", "://data."),
            "utm": "?utm_source=qgis_plugin&utm_medium=about&utm_campaign="
            + self.__package_name,
        }

    def __html(self, metadata: Dict[str, Optional[str]]) -> str:
        report_end = self.tr("REPORT_END")
        if report_end == "REPORT_END":
            report_end = ""

        titles = {
            "developers_title": self.tr("Developers"),
            "homepage_title": self.tr("Homepage"),
            "report_title": self.tr("Please report bugs at"),
            "report_end": report_end,
            "bugtracker_title": self.tr("bugtracker"),
            "video_title": self.tr("Video with an overview of the plugin"),
            "services_title": self.tr("Other helpful services by NextGIS"),
            "extracts_title": self.tr(
                "Convenient up-to-date data extracts for any place in the world"
            ),
            "webgis_title": self.tr("Fully featured Web GIS service"),
        }

        description = """
            <p>{description}</p>
            <p>{about}</p>
            <p><b>{developers_title}:</b> <a href="{main_url}/{utm}">{authors}</a></p>
            <p><b>{homepage_title}:</b> <a href="{homepage_url}">{homepage_url}</a></p>
            <p><b>{report_title}</b> <a href="{tracker_url}">{bugtracker_title}</a> {report_end}</p>
            """

        if metadata.get("video_url") is not None:
            description += '<p><b>{video_title}:</b> <a href="{video_url}">{video_url}</a></p>'

        services = """
            <p>
            {services_title}:
            <ul>
              <li><b>{extracts_title}</b>: <a href="{data_url}/{utm}">{data_url}</a></li>
              <li><b>{webgis_title}</b>: <a href="{main_url}/nextgis-com/plans{utm}">{main_url}/nextgis-com/plans</a></li>
            </ul>
            </p>
            """

        replacements = dict()
        replacements.update(titles)
        replacements.update(metadata)

        return (description + services).format_map(replacements)
