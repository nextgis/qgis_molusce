#!/bin/bash
pyrcc5 -o resources_rc.py resources.qrc
pyuic5 -o ui/ui_aboutdialogbase.py ui/aboutdialogbase.ui
pyuic5 -o ui/ui_logisticregressionwidgetbase.py ui/logisticregressionwidgetbase.ui
pyuic5 -o ui/ui_moluscedialogbase.py ui/moluscedialogbase.ui
pyuic5 -o ui/ui_multicriteriaevaluationwidgetbase.py ui/multicriteriaevaluationwidgetbase.ui
pyuic5 -o ui/ui_neuralnetworkwidgetbase.py ui/neuralnetworkwidgetbase.ui
pyuic5 -o ui/ui_weightofevidencewidgetbase.py ui/weightofevidencewidgetbase.ui