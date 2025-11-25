# Details

Date : 2025-11-25 23:46:17

Directory d:\\code\\math-content-recognition-

Total : 70 files,  5125 codes, 769 comments, 1158 blanks, all 7052 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [.claude/agents/commit-crafter.md](/.claude/agents/commit-crafter.md) | Markdown | 146 | 0 | 19 | 165 |
| [.claude/agents/staged-code-reviewer.md](/.claude/agents/staged-code-reviewer.md) | Markdown | 50 | 0 | 22 | 72 |
| [.claude/commands/code-review.md](/.claude/commands/code-review.md) | Markdown | 1 | 0 | 1 | 2 |
| [.claude/commands/fix-github-issue.md](/.claude/commands/fix-github-issue.md) | Markdown | 10 | 0 | 4 | 14 |
| [.claude/commands/make-commit.md](/.claude/commands/make-commit.md) | Markdown | 12 | 0 | 5 | 17 |
| [.github/workflows/deploy-doc.yml](/.github/workflows/deploy-doc.yml) | YAML | 29 | 0 | 3 | 32 |
| [.github/workflows/pr-welcome.yml](/.github/workflows/pr-welcome.yml) | YAML | 33 | 0 | 9 | 42 |
| [.github/workflows/publish.yml](/.github/workflows/publish.yml) | YAML | 27 | 0 | 8 | 35 |
| [.github/workflows/python-lint.yml](/.github/workflows/python-lint.yml) | YAML | 22 | 0 | 6 | 28 |
| [.github/workflows/test.yaml](/.github/workflows/test.yaml) | YAML | 28 | 0 | 8 | 36 |
| [.pre-commit-config.yaml](/.pre-commit-config.yaml) | YAML | 22 | 0 | 2 | 24 |
| [README.md](/README.md) | Markdown | 151 | 1 | 76 | 228 |
| [assets/README\_zh.md](/assets/README_zh.md) | Markdown | 146 | 1 | 73 | 220 |
| [assets/fire.svg](/assets/fire.svg) | XML | 460 | 0 | 1 | 461 |
| [assets/logo.svg](/assets/logo.svg) | XML | 11 | 0 | 4 | 15 |
| [docs/Makefile](/docs/Makefile) | Makefile | 9 | 7 | 5 | 21 |
| [docs/make.bat](/docs/make.bat) | Batch | 26 | 1 | 9 | 36 |
| [docs/requirements.txt](/docs/requirements.txt) | pip requirements | 0 | 0 | 1 | 1 |
| [docs/source/api.rst](/docs/source/api.rst) | reStructuredText | 15 | 10 | 15 | 40 |
| [docs/source/conf.py](/docs/source/conf.py) | Python | 46 | 16 | 14 | 76 |
| [docs/source/index.rst](/docs/source/index.rst) | reStructuredText | 50 | 5 | 23 | 78 |
| [examples/client\_demo.py](/examples/client_demo.py) | Python | 7 | 0 | 4 | 11 |
| [examples/train\_texteller/dataset/train/metadata.jsonl](/examples/train_texteller/dataset/train/metadata.jsonl) | JSON Lines | 35 | 0 | 1 | 36 |
| [examples/train\_texteller/train.py](/examples/train_texteller/train.py) | Python | 51 | 6 | 15 | 72 |
| [examples/train\_texteller/train\_config.yaml](/examples/train_texteller/train_config.yaml) | YAML | 30 | 1 | 2 | 33 |
| [examples/train\_texteller/utils/\_\_init\_\_.py](/examples/train_texteller/utils/__init__.py) | Python | 16 | 0 | 2 | 18 |
| [examples/train\_texteller/utils/augraphy\_pipe.py](/examples/train_texteller/utils/augraphy_pipe.py) | Python | 140 | 17 | 9 | 166 |
| [examples/train\_texteller/utils/functional.py](/examples/train_texteller/utils/functional.py) | Python | 32 | 1 | 15 | 48 |
| [examples/train\_texteller/utils/transforms.py](/examples/train_texteller/utils/transforms.py) | Python | 105 | 10 | 40 | 155 |
| [tests/test\_globals.py](/tests/test_globals.py) | Python | 34 | 14 | 20 | 68 |
| [tests/test\_to\_katex.py](/tests/test_to_katex.py) | Python | 5 | 1 | 3 | 9 |
| [texteller/\_\_init\_\_.py](/texteller/__init__.py) | Python | 3 | 0 | 2 | 5 |
| [texteller/api/\_\_init\_\_.py](/texteller/api/__init__.py) | Python | 23 | 0 | 2 | 25 |
| [texteller/api/criterias/\_\_init\_\_.py](/texteller/api/criterias/__init__.py) | Python | 2 | 0 | 3 | 5 |
| [texteller/api/criterias/ngram.py](/texteller/api/criterias/ngram.py) | Python | 22 | 32 | 10 | 64 |
| [texteller/api/detection/\_\_init\_\_.py](/texteller/api/detection/__init__.py) | Python | 2 | 0 | 2 | 4 |
| [texteller/api/detection/detect.py](/texteller/api/detection/detect.py) | Python | 40 | 21 | 9 | 70 |
| [texteller/api/detection/preprocess.py](/texteller/api/detection/preprocess.py) | Python | 91 | 49 | 22 | 162 |
| [texteller/api/format.py](/texteller/api/format.py) | Python | 431 | 88 | 135 | 654 |
| [texteller/api/inference.py](/texteller/api/inference.py) | Python | 174 | 67 | 17 | 258 |
| [texteller/api/katex.py](/texteller/api/katex.py) | Python | 89 | 24 | 20 | 133 |
| [texteller/api/load.py](/texteller/api/load.py) | Python | 49 | 67 | 9 | 125 |
| [texteller/cli/\_\_init\_\_.py](/texteller/cli/__init__.py) | Python | 13 | 3 | 8 | 24 |
| [texteller/cli/commands/\_\_init\_\_.py](/texteller/cli/commands/__init__.py) | Python | 0 | 3 | 1 | 4 |
| [texteller/cli/commands/inference.py](/texteller/cli/commands/inference.py) | Python | 39 | 2 | 3 | 44 |
| [texteller/cli/commands/launch/\_\_init\_\_.py](/texteller/cli/commands/launch/__init__.py) | Python | 93 | 2 | 7 | 102 |
| [texteller/cli/commands/launch/server.py](/texteller/cli/commands/launch/server.py) | Python | 58 | 0 | 8 | 66 |
| [texteller/cli/commands/web/\_\_init\_\_.py](/texteller/cli/commands/web/__init__.py) | Python | 6 | 1 | 3 | 10 |
| [texteller/cli/commands/web/streamlit\_demo.py](/texteller/cli/commands/web/streamlit_demo.py) | Python | 163 | 6 | 48 | 217 |
| [texteller/cli/commands/web/style.py](/texteller/cli/commands/web/style.py) | Python | 49 | 0 | 7 | 56 |
| [texteller/constants.py](/texteller/constants.py) | Python | 19 | 7 | 8 | 34 |
| [texteller/globals.py](/texteller/globals.py) | Python | 17 | 18 | 7 | 42 |
| [texteller/logger.py](/texteller/logger.py) | Python | 55 | 23 | 19 | 97 |
| [texteller/models/\_\_init\_\_.py](/texteller/models/__init__.py) | Python | 2 | 0 | 2 | 4 |
| [texteller/models/texteller.py](/texteller/models/texteller.py) | Python | 40 | 0 | 8 | 48 |
| [texteller/paddleocr/CTCLabelDecode.py](/texteller/paddleocr/CTCLabelDecode.py) | Python | 166 | 17 | 30 | 213 |
| [texteller/paddleocr/DBPostProcess.py](/texteller/paddleocr/DBPostProcess.py) | Python | 169 | 17 | 36 | 222 |
| [texteller/paddleocr/operators.py](/texteller/paddleocr/operators.py) | Python | 143 | 13 | 31 | 187 |
| [texteller/paddleocr/predict\_det.py](/texteller/paddleocr/predict_det.py) | Python | 234 | 17 | 27 | 278 |
| [texteller/paddleocr/predict\_rec.py](/texteller/paddleocr/predict_rec.py) | Python | 315 | 20 | 45 | 380 |
| [texteller/paddleocr/utility.py](/texteller/paddleocr/utility.py) | Python | 490 | 104 | 96 | 690 |
| [texteller/types/\_\_init\_\_.py](/texteller/types/__init__.py) | Python | 6 | 0 | 7 | 13 |
| [texteller/types/bbox.py](/texteller/types/bbox.py) | Python | 42 | 3 | 12 | 57 |
| [texteller/utils/\_\_init\_\_.py](/texteller/utils/__init__.py) | Python | 25 | 0 | 2 | 27 |
| [texteller/utils/bbox.py](/texteller/utils/bbox.py) | Python | 114 | 4 | 23 | 141 |
| [texteller/utils/device.py](/texteller/utils/device.py) | Python | 22 | 10 | 10 | 42 |
| [texteller/utils/image.py](/texteller/utils/image.py) | Python | 82 | 15 | 25 | 122 |
| [texteller/utils/latex.py](/texteller/utils/latex.py) | Python | 53 | 39 | 24 | 116 |
| [texteller/utils/misc.py](/texteller/utils/misc.py) | Python | 3 | 0 | 3 | 6 |
| [texteller/utils/path.py](/texteller/utils/path.py) | Python | 32 | 6 | 8 | 46 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)