# PassageNote — 영어 원문·문제 사진 풀이 학습 노트

수능·모의고사·내신 영어 객관식 문제, 한국어/한·영 혼합 서술형 영작, 원문 독해를 사진(최대 6장) 또는 텍스트로 분석하고 1:1 AI 튜터와 학습할 수 있는 웹 앱입니다.

## Vercel 배포 방법 (권장)

1. 이 폴더의 파일들을 GitHub 저장소에 업로드합니다.
2. Vercel에서 해당 GitHub 저장소를 Import 합니다. (`vercel.json`과 `api/index.ts`가 자동으로 인식되며, npm `cache-dir` 충돌 방지 설정이 포함되어 있습니다.)
3. **Project Settings → Environment Variables**에 `GEMINI_API_KEY` 값을 추가합니다.
4. **Deploy**를 누르면 프론트엔드(Vite)와 백엔드 API(`/api/*`)가 함께 배포됩니다.
5. 만약 이전 배포 캐시가 남아 오류가 난다면 Vercel **Deployments → Redeploy** 시 **Use existing Build Cache** 체크를 해제하고 재배포해 주세요.

## 로컬 실행 방법

```bash
npm install
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 값을 입력하세요
npm run dev
```

## 프로덕션 빌드 및 실행

```bash
npm run build
npm start
```

> `.env` 등 실제 API 키 비밀 파일은 보안을 위해 압축 파일에 포함되지 않습니다.
