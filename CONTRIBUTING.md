# Contributing Guide

본 문서는 프로젝트에 참여하는 모든 개발자가 일관된 방식으로 협업하기 위한 가이드입니다.

모든 팀원은 코드 작성 및 변경사항 제출 시 아래 규칙을 준수해야 합니다.

## 1. Branch Strategy

본 프로젝트는 **간소화된 Git Flow**를 사용합니다.

### 1.1 Branch Roles

| Branch | Description |
|---|---|
| `main` | 배포 가능한 안정 버전 |
| `dev` | 기능 통합 및 검증을 위한 개발 브랜치 |
| `feature/*` | 신규 기능 개발 |
| `fix/*` | 일반적인 버그 수정 |
| `hotfix/*` | 배포 버전의 긴급 오류 수정 |

### 1.2 Branch Workflow

기본적인 개발 흐름은 다음과 같습니다.

1. `dev`에서 새로운 기능 브랜치를 생성합니다.
2. 기능 브랜치에서 개발 및 테스트를 진행합니다.
3. 작업이 완료되면 `dev`를 대상으로 Pull Request를 생성합니다.
4. 코드 리뷰 및 검증을 통과하면 `dev`에 병합합니다.
5. 배포 준비가 완료되면 `dev`에서 `main`으로 Pull Request를 생성합니다.
6. 병합이 완료된 기능 브랜치는 삭제합니다.

**Branch Protection**

- `main` 및 `dev` 브랜치에 직접 Push하지 않습니다.
- 모든 변경사항은 Pull Request를 통해 반영합니다.
- `main`에는 배포 가능한 상태의 코드만 병합합니다.
- `dev`에는 기본적인 실행 및 테스트가 가능한 코드만 병합합니다.
- 미완성 기능은 기본적으로 병합하지 않습니다. 단, 안정적으로 비활성화된 기능은 예외로 허용할 수 있습니다.
- 공유 브랜치에서는 Force Push를 금지합니다.

**Hotfix Workflow**

배포 중인 `main`에서 긴급한 수정이 필요한 경우 다음 절차를 따릅니다.

1. `main`에서 `hotfix/*` 브랜치를 생성합니다.
2. 오류를 수정하고 검증합니다.
3. Pull Request를 통해 `main`에 병합합니다.
4. 동일한 수정사항을 `dev`에도 반영합니다.

### 1.3 Branch Naming Convention

브랜치 이름은 다음 형식을 따릅니다.

`<type>/<short-description>`

Issue 번호가 존재하면 다음 형식을 권장합니다.

`<type>/<issue-number>-<short-description>`

**Examples**

- `feature/42-document-chunking`
- `feature/rag-pipeline`
- `feature/login-api`
- `fix/53-token-expiration`
- `hotfix/server-startup`

**Naming Rules**

- 영문 소문자를 사용합니다.
- 단어는 하이픈(`-`)으로 구분합니다.
- 브랜치 이름은 작업 목적을 명확하게 표현해야 합니다.
- 서로 독립적인 기능은 가능한 한 별도 브랜치에서 개발합니다.

---

## 2. Commit Convention

본 프로젝트는 **Conventional Commits** 규칙을 사용합니다.

### 2.1 Commit Message Format

`<type>(<scope>): <description>`

- `type`: 변경사항의 종류
- `scope`: 변경 대상 모듈 또는 기능 (선택 사항)
- `description`: 변경사항에 대한 간결한 설명

### 2.2 Commit Types

| Type | Description |
|---|---|
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변경 없는 코드 구조 개선 |
| `docs` | 문서 추가 및 수정 |
| `test` | 테스트 코드 추가 및 수정 |
| `chore` | 설정, 의존성 및 유지보수 작업 |
| `style` | 코드 포맷팅 변경 |
| `perf` | 성능 개선 |
| `ci` | CI/CD 설정 변경 |
| `build` | 빌드 시스템 및 빌드 의존성 변경 |
| `revert` | 이전 변경사항 되돌리기 |

### 2.3 Commit Message Examples

- `feat(rag): implement document chunking`
- `feat(api): add authentication endpoint`
- `fix(parser): handle empty document`
- `refactor(agent): simplify state management`
- `docs: update installation guide`
- `test(rag): add retrieval unit tests`
- `chore: update dependencies`

### 2.4 Commit Guidelines

- 커밋 메시지는 영어로 작성합니다.
- `type`과 `scope`는 소문자로 작성합니다.
- 제목은 명령형 동사로 시작하는 간결한 표현을 권장합니다.
- 제목 끝에 마침표를 사용하지 않습니다.
- 하나의 커밋에는 가능한 한 하나의 논리적인 변경사항만 포함합니다.
- `update`, `fix stuff`, `작업` 등 변경 목적이 불명확한 메시지는 지양합니다.
- 코드 변경과 무관한 파일은 가능한 한 같은 커밋에 포함하지 않습니다.

---

## 3. Pull Request Guidelines

모든 기능 추가 및 코드 수정은 Pull Request를 통해 병합합니다.

### 3.1 PR Naming Convention

PR 제목에도 Conventional Commits 형식을 사용합니다.

**Examples**

- `feat(rag): implement vector search`
- `fix(api): resolve response timeout`
- `refactor(frontend): separate UI components`

### 3.2 PR Description

모든 PR에는 다음 내용을 작성합니다.

**Summary**

변경 목적과 주요 구현 내용을 설명합니다.

**Changes**

추가, 수정, 삭제한 주요 기능을 설명합니다.

**Testing**

수행한 테스트 및 검증 결과를 기록합니다.

**Related Issues**

관련 GitHub Issue 번호를 연결합니다.

권장 PR 템플릿:

**Summary**
- 

**Changes**
- 

**Testing**
- [ ] 로컬 실행 확인
- [ ] 관련 테스트 통과
- [ ] 기존 기능 영향 확인

**Related Issues**
- 

### 3.3 Review Policy

- `dev` 병합에는 최소 1명의 다른 팀원 승인이 필요합니다.
- PR 작성자는 병합 전 변경사항을 자체 검토해야 합니다.
- 리뷰에서 요청된 수정사항은 병합 전에 처리합니다.
- 리뷰어는 기능의 정확성, 가독성 및 기존 코드와의 호환성을 확인합니다.
- 가능한 한 하나의 PR은 하나의 기능 또는 작업 단위로 구성합니다.

### 3.4 Merge Policy

기능 브랜치에서 `dev`로 병합할 때는 기본적으로 **Squash and Merge** 방식을 사용합니다.

- 여러 작업 커밋을 하나의 논리적인 변경 기록으로 정리합니다.
- Squash Commit 메시지는 커밋 규칙을 따릅니다.
- 병합 전 충돌이 발생했다면 해결하고 다시 검증합니다.
- 병합이 완료된 기능 브랜치는 삭제합니다.

`dev`에서 `main`으로 병합할 때는 배포 단위의 기록을 보존하는 방식을 사용하며, 팀에서 정한 릴리스 정책을 따릅니다.

---

## 4. Issue Management

GitHub Issues를 사용하여 개발 작업 및 버그를 관리합니다.

### 4.1 Issue Categories

| Label | Description |
|---|---|
| `feature` | 새로운 기능 구현 |
| `bug` | 오류 및 버그 |
| `enhancement` | 기존 기능 개선 |
| `documentation` | 문서 작업 |
| `refactor` | 코드 구조 개선 |
| `question` | 논의 또는 질문 |

### 4.2 Issue Guidelines

- 가능한 한 기능 개발 전에 Issue를 생성합니다.
- Issue에는 작업 목적, 요구사항, 완료 조건을 명확히 작성합니다.
- 규모가 큰 작업은 독립적인 작은 작업으로 분리하는 것을 권장합니다.
- 관련 PR에서 Issue 번호를 참조합니다.
- Issue와 PR을 연결하여 작업 진행 상황을 추적합니다.

---

## 5. Code Style & Formatting

코드 일관성을 유지하기 위해 언어 및 프레임워크별 코딩 규칙을 준수합니다.

### 5.1 General Guidelines

- 변수, 함수, 클래스 이름은 역할을 명확하게 표현해야 합니다.
- 불필요한 중복 코드를 지양합니다.
- 하나의 함수와 클래스는 가능한 한 명확한 책임을 갖도록 설계합니다.
- 복잡한 알고리즘이나 의도가 불분명한 로직에는 주석을 작성합니다.
- 사용하지 않는 코드, 임시 디버깅 출력 및 불필요한 주석은 제거합니다.
- 기존 프로젝트의 디렉터리 구조 및 네이밍 규칙을 따릅니다.

### 5.2 Formatting

언어별 포맷터와 린터를 우선적으로 사용합니다.

새로운 도구나 설정을 도입할 경우 팀원들과 논의하고 동일한 설정을 공유합니다.

개인적인 코드 스타일보다 프로젝트에서 합의한 규칙을 우선합니다.

---

## 6. Testing Guidelines

모든 PR은 병합 전에 변경사항이 정상적으로 동작하는지 검증해야 합니다.

### 6.1 Required Checks

- [ ] 변경한 기능이 의도대로 동작하는가?
- [ ] 기존 기능에 영향을 주지 않는가?
- [ ] 관련 테스트가 통과하는가?
- [ ] 오류 및 예외 상황을 적절하게 처리하는가?
- [ ] 디버깅 코드와 불필요한 파일을 제거했는가?
- [ ] 필요한 문서를 수정했는가?

### 6.2 Testing Policy

- 새로운 기능에는 가능한 한 관련 테스트를 추가합니다.
- 버그 수정 시 동일한 문제가 재발하지 않도록 회귀 테스트를 권장합니다.
- CI가 구성되어 있다면 필수 검사 통과 후 병합합니다.
- 자동화된 테스트가 없는 부분은 수동 검증 결과를 PR에 작성합니다.
- 테스트를 통과하지 못한 변경사항은 원칙적으로 `dev`에 병합하지 않습니다.

---

## 7. Collaboration Rules

### 7.1 Development Environment

- 개발 환경 설정 및 실행 방법은 `README.md`를 참고합니다.
- 새로운 의존성을 추가하면 관련 설정 파일과 문서를 업데이트합니다.
- 팀원 간 환경 차이로 문제가 발생하지 않도록 주요 버전과 설정을 공유합니다.

### 7.2 Security

- API Key, Access Token, Password 등의 인증정보를 저장소에 커밋하지 않습니다.
- 로컬 환경변수는 `.env` 등의 파일로 관리하고 `.gitignore`에 등록합니다.
- 필요한 환경변수는 실제 비밀값이 없는 `.env.example` 파일로 공유하는 것을 권장합니다.
- 비밀정보가 실수로 커밋되었다면 즉시 팀에 알리고 해당 인증정보를 폐기하거나 재발급합니다.

### 7.3 Conflict Resolution

- 기능 브랜치를 장기간 유지할 경우 `dev`의 최신 변경사항을 주기적으로 반영합니다.
- Merge Conflict가 발생하면 변경 의도를 확인한 뒤 해결합니다.
- 다른 팀원이 작성한 코드를 임의로 삭제하거나 변경하지 않습니다.
- 충돌 해결 후 관련 기능을 다시 검증합니다.

### 7.4 Communication

- 다른 기능이나 공통 인터페이스에 영향을 주는 변경사항은 사전에 공유합니다.
- API 규격, 데이터 구조, 디렉터리 구조 등 공통 설계 변경은 팀원과 논의합니다.
- 중요한 기술적 결정은 Issue, PR 또는 별도 문서로 기록합니다.
- 구현에 문제가 발생하거나 일정에 영향을 줄 가능성이 있다면 팀에 공유합니다.

---

## 8. Definition of Done

기능 개발은 다음 조건을 만족할 때 완료된 것으로 간주합니다.

- [ ] 요구사항을 충족하는 기능 구현
- [ ] 로컬 환경에서 정상 동작 확인
- [ ] 관련 테스트 및 검증 완료
- [ ] 기존 기능과의 호환성 확인
- [ ] 코드 스타일 및 커밋 규칙 준수
- [ ] 필요한 문서 업데이트
- [ ] Pull Request 리뷰 승인
- [ ] `dev` 브랜치 병합 완료

단, 미완성 기능이 다른 기능에 영향을 주지 않도록 명확하게 비활성화되어 있으며 팀에서 승인한 경우에는 예외를 허용할 수 있습니다.

---

본 가이드는 프로젝트 진행 과정에서 팀원들의 합의를 통해 수정할 수 있습니다. 규칙의 변경은 Issue 또는 Pull Request를 통해 제안하고 논의합니다.
