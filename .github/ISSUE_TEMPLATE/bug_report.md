---
name: Bug report
description: Report a problem or regression
labels: [bug]
body:
  - type: markdown
    attributes:
      value: "Thanks for reporting. Please include enough detail to reproduce."
  - type: input
    id: environment
    attributes:
      label: Environment
      description: OS, Python version, and dependency versions
      placeholder: "Windows 11, Python 3.11, pandas 2.x"
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      description: Exact commands and inputs
      placeholder: "1. Run python main.py\n2. ..."
    validations:
      required: true
  - type: textarea
    id: expected
    attributes:
      label: Expected behavior
    validations:
      required: true
  - type: textarea
    id: actual
    attributes:
      label: Actual behavior
    validations:
      required: true
  - type: textarea
    id: logs
    attributes:
      label: Logs or screenshots
      description: Paste logs or add screenshots if relevant
      render: shell
    validations:
      required: false
