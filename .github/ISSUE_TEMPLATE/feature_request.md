---
name: Feature request
description: Suggest an improvement
labels: [enhancement]
body:
  - type: markdown
    attributes:
      value: "Thanks for the suggestion. Please be specific."
  - type: textarea
    id: problem
    attributes:
      label: Problem statement
      description: What problem does this solve?
    validations:
      required: true
  - type: textarea
    id: proposal
    attributes:
      label: Proposed solution
      description: Describe the desired change
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: Any other approaches?
    validations:
      required: false
