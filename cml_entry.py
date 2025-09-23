from forcateri.reporting.clearmlreporter.clearmlreporter import ClearMLReporter
from src import project_root

cml_reporter = ClearMLReporter(config_name='pipeline', project_root=project_root)
