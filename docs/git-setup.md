# GitHub 저장소 설정 가이드

이 가이드는 로컬 Git 저장소를 GitHub에 연결하는 방법을 설명합니다.

## 현재 상태

✅ 로컬 git 저장소 초기화 완료
✅ .gitignore 파일 생성 완료
✅ 초기 커밋 완료

## 다음 단계: GitHub에 연결하기

### 1. GitHub에서 새 저장소 생성

1. <https://github.com/new> 로 이동합니다
2. 저장소 이름을 입력합니다 (예: `dx-to-ax`)
3. 설명을 추가합니다 (선택사항)
4. Public 또는 Private을 선택합니다
5. **중요: README, .gitignore, 라이선스로 초기화하지 마세요** (이미 로컬에 있습니다)
6. "Create repository" 버튼을 클릭합니다

### 2. 로컬 저장소를 GitHub에 연결

GitHub에서 저장소를 생성한 후, 다음 명령어 중 하나를 실행합니다:

#### 옵션 A: HTTPS (대부분의 사용자에게 권장)

```bash
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

#### 옵션 B: SSH (SSH 키가 설정된 경우)

```bash
git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git
git push -u origin main
```

**`YOUR_USERNAME`과 `REPO_NAME`을 실제 GitHub 사용자명과 저장소 이름으로 변경하세요.**

### 3. 연결 확인

푸시 후 연결을 확인합니다:

```bash
git remote -v
```

다음과 같이 표시되어야 합니다:

```text
origin  https://github.com/YOUR_USERNAME/REPO_NAME.git (fetch)
origin  https://github.com/YOUR_USERNAME/REPO_NAME.git (push)
```

## 문제 해결

### 인증 문제

인증 문제가 발생하는 경우:

- **HTTPS**: 비밀번호 대신 Personal Access Token을 사용해야 할 수 있습니다
  - 토큰 생성: <https://github.com/settings/tokens>
  - 프롬프트가 나타나면 토큰을 비밀번호로 사용하세요

- **SSH**: SSH 키가 올바르게 설정되어 있는지 확인하세요
  - 확인: `ssh -T git@github.com`
  - 설정 가이드: <https://docs.github.com/en/authentication/connecting-to-github-with-ssh>

### 잘못된 저장소에 푸시한 경우

원격 URL을 변경해야 하는 경우:

```bash
git remote set-url origin NEW_URL
```

## 이후 커밋

초기 설정 후에는 다음 명령어를 사용하여 정기적으로 업데이트합니다:

```bash
git add .
git commit -m "커밋 메시지"
git push
```
