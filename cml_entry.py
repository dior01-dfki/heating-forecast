from forcateri.reporting.clearmlreporter.clearmlreporter import ClearMLReporter
from src import project_root

cml_reporter = ClearMLReporter(config_name='pipeline', project_root=project_root)
print(cml_reporter.config)
print("\n\n\n")
print(cml_reporter.args)
cml_reporter.args.append(('config', 'pipeline'))
cml_reporter.execute_task_enq()