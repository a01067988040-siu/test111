import 'dotenv/config';
import express, { type Request, type Response } from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import zlib from 'zlib';
import { GoogleGenAI, ThinkingLevel, Type } from '@google/genai';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const PRIMARY_MODEL = 'gemini-4-hydrogen';
export const FALLBACK_MODEL = 'gemini-3.1-flash-lite';

function resolveProjectRoot(): string {
  const candidates = [
    path.resolve(__dirname, '..'),
    path.resolve(__dirname),
    process.cwd(),
  ];
  for (const dir of candidates) {
    if (fs.existsSync(path.join(dir, 'package.json'))) {
      return dir;
    }
  }
  return process.cwd();
}

function getAiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error(
      'GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다. AI Studio Secrets 또는 Vercel Project Settings > Environment Variables에서 GEMINI_API_KEY를 등록해 주세요.'
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function isTransientUpstreamError(err: unknown): boolean {
  const msg = (err instanceof Error ? err.message : String(err)).toLowerCase();
  return (
    msg.includes('503') ||
    msg.includes('504') ||
    msg.includes('deadline') ||
    msg.includes('unavailable') ||
    msg.includes('high demand') ||
    msg.includes('429') ||
    msg.includes('resource_exhausted') ||
    msg.includes('timeout') ||
    msg.includes('aborted') ||
    msg.includes('500') ||
    msg.includes('internal') ||
    msg.includes('fetch failed')
  );
}

async function generateWithFallback(
  ai: GoogleGenAI,
  contents: unknown,
  config: Record<string, unknown>,
  primaryTimeoutMs = 18000,
  primaryThinking: ThinkingLevel = ThinkingLevel.MEDIUM
) {
  try {
    return await ai.models.generateContent({
      model: PRIMARY_MODEL,
      contents: contents as never,
      config: {
        ...config,
        thinkingConfig: { thinkingLevel: primaryThinking },
        httpOptions: { timeout: primaryTimeoutMs },
      },
    });
  } catch (primaryErr) {
    if (!isTransientUpstreamError(primaryErr)) {
      throw primaryErr;
    }
    console.warn('Primary model unavailable, switching to fallback:', FALLBACK_MODEL);
    return await ai.models.generateContent({
      model: FALLBACK_MODEL,
      contents: contents as never,
      config: {
        ...config,
        thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
        httpOptions: { timeout: 45000 },
      },
    });
  }
}

const analysisSchema = {
  type: Type.OBJECT,
  properties: {
    title: {
      type: Type.STRING,
      description: '지문 또는 문제의 핵심 주제를 요약한 간결한 한국어 제목 (20자 이내 권장)',
    },
    languageFormat: {
      type: Type.STRING,
      description:
        '문제의 언어 구성: 주로 영어 지문·선지면 "english", 주로 한국어 지시문·제시문 중심의 영작/개념 문제면 "korean", 영어 지문과 한국어 발문/선지/조건이 함께 있으면 "mixed"',
    },
    detectedPostType: {
      type: Type.STRING,
      description:
        '객관식 또는 단답형 문제 풀이면 "problem", 한국어 제시문·조건 기반 영작/빈칸 완성 서술형 문제면 "writing", 문제 없이 순수 영어 원문 독해면 "passage"',
    },
    extractedText: {
      type: Type.STRING,
      description:
        '이미지에서 추출한 전체 원문 텍스트(한국어 발문, <보기>, <조건>, 영어 지문, ①~⑤ 선택지 모두 포함, 줄바꿈 유지)',
    },
    koreanInstruction: {
      type: Type.STRING,
      description: '한국어로 된 문제 발문, 지시문 또는 우리말 해석 제시문. 없으면 빈 문자열.',
    },
    givenWords: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '<보기>에 주어진 단어·어구 목록. 없으면 빈 배열.',
    },
    conditions: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description:
        '<조건> 항목 목록 (예: "관계대명사 what을 포함할 것", "필요시 어형을 변형할 것"). 없으면 빈 배열.',
    },
    summaryKorean: {
      type: Type.STRING,
      description: '지문 또는 문제의 핵심 요지와 학습 포인트를 1~2문장의 명료한 한국어로 요약',
    },
    isProblem: {
      type: Type.BOOLEAN,
      description: '객관식(①~⑤) 또는 선택형 문제 풀이가 포함되어 있으면 true',
    },
    questionType: {
      type: Type.STRING,
      description:
        '문제 유형 (예: 빈칸 추론, 어법성 판단, 주제/제목, 내용 일치, 어휘 추론, 영작 서술형, 원문 독해)',
    },
    correctAnswer: {
      type: Type.STRING,
      description: '객관식 정답 번호(예: "①", "③") 또는 단답형 정답. 객관식이 아니면 빈 문자열.',
    },
    answerSummary: {
      type: Type.STRING,
      description: '정답이 도출되는 핵심 논리적 근거와 풀이 과정 (한국어 2~4문장). 객관식이 아니면 빈 문자열.',
    },
    evidenceSentences: {
      type: Type.ARRAY,
      items: { type: Type.INTEGER },
      description: '정답 또는 영작의 결정적 단서가 되는 sentences 배열의 문장 index 목록 (1-based). 없으면 빈 배열.',
    },
    choices: {
      type: Type.ARRAY,
      description:
        '객관식 선택지(①~⑤)가 있는 경우 각 선지별 정오답 해설(영어 선지, 한국어 선지 모두 지원). 없으면 빈 배열.',
      items: {
        type: Type.OBJECT,
        properties: {
          label: { type: Type.STRING, description: '선지 번호 (예: ①, ②, ③, ④, ⑤)' },
          text: { type: Type.STRING, description: '선지 내용 (영어 또는 한국어 원문 그대로)' },
          isCorrect: { type: Type.BOOLEAN, description: '정답 여부' },
          explanation: { type: Type.STRING, description: '이 선지가 정답이거나 오답인 구체적인 이유 (한국어)' },
        },
        required: ['label', 'text', 'isCorrect', 'explanation'],
      },
    },
    isWriting: {
      type: Type.BOOLEAN,
      description: '영작, 단어 배열, 빈칸 서술형 완성, 한·영 혼합 서술형 문제가 포함되어 있으면 true',
    },
    writingType: {
      type: Type.STRING,
      description:
        '영작 세부 유형 (예: 조건부 영작, 주어진 단어 배열, 한·영 혼합 서술형, 요약문 빈칸 영작). 아니면 빈 문자열.',
    },
    writingKoreanPrompt: {
      type: Type.STRING,
      description: '영작해야 할 우리말 뜻 또는 서술형 요구사항 요약. 아니면 빈 문자열.',
    },
    writingContextTemplate: {
      type: Type.STRING,
      description:
        '영어 문장 속 일부 빈칸이나 한국어 괄호를 채우는 한·영 혼합 서술형인 경우 앞뒤 영어 문맥 틀 (예: "The study shows that [ ________ ] is vital."). 전체 문장 영작이면 빈 문자열.',
    },
    writingModelAnswer: {
      type: Type.STRING,
      description: '조건을 완벽히 충족하는 모범 영어 정답 (문장 또는 빈칸 정답). 영작이 아니면 빈 문자열.',
    },
    writingAlternativeAnswers: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '문법적·의미적으로 정답 인정 가능한 대체 영어 표현 0~2개.',
    },
    writingSteps: {
      type: Type.ARRAY,
      description: '영작 문제일 경우 주어·동사 뼈대 잡기부터 조건 적용까지 4단계 빌드업 해설. 영작이 아니면 빈 배열.',
      items: {
        type: Type.OBJECT,
        properties: {
          stepNumber: { type: Type.INTEGER, description: '단계 번호 (1~4)' },
          title: { type: Type.STRING, description: '단계 제목 (예: 1단계: 주어·동사 핵심 뼈대 잡기)' },
          englishDraft: { type: Type.STRING, description: '해당 단계에서 완성된 영어 어구 또는 문장 초안' },
          explanation: { type: Type.STRING, description: '어순 배치, 어형 변화, 조건 적용 이유를 설명하는 한국어 해설' },
        },
        required: ['stepNumber', 'title', 'englishDraft', 'explanation'],
      },
    },
    writingDeductionPoints: {
      type: Type.ARRAY,
      items: { type: Type.STRING },
      description: '내신 서술형 채점 시 학생들이 자주 틀려 감점되는 주의 사항 2~4개.',
    },
    sentences: {
      type: Type.ARRAY,
      description:
        '영어 지문 문장(영어/한영 혼합 문제) 또는 완성된 영어 모범 답안 문장(단독 영작 문제)의 순서별 직독직해·구문 분석 목록. 이미지에서 텍스트를 읽을 수 없으면 빈 배열.',
      items: {
        type: Type.OBJECT,
        properties: {
          index: { type: Type.INTEGER, description: '1부터 시작하는 문장 번호' },
          original: { type: Type.STRING, description: '원문 영어 문장 (또는 모범 영작 문장)' },
          chunked: {
            type: Type.STRING,
            description: '의미 단위(청크)마다 " / "로 끊어 표기한 직독직해용 영어 문장',
          },
          literalTranslation: {
            type: Type.STRING,
            description: 'chunked의 "/" 끊기 순서에 1:1로 대응되도록 번역한 한국어 직독직해',
          },
          naturalTranslation: {
            type: Type.STRING,
            description: '문맥을 살린 자연스러운 한국어 의역',
          },
          grammarPoints: {
            type: Type.ARRAY,
            description: '수능·내신에서 중요한 핵심 구문/문법 포인트 (핵심 문장 위주 0~2개, 간결하게)',
            items: {
              type: Type.OBJECT,
              properties: {
                pointTitle: { type: Type.STRING, description: '문법/구문 명칭' },
                targetPhrase: { type: Type.STRING, description: '문장 내 해당 영어 어구' },
                explanation: { type: Type.STRING, description: '중고등학생 눈높이의 정확한 문법·해석 설명' },
              },
              required: ['pointTitle', 'targetPhrase', 'explanation'],
            },
          },
        },
        required: ['index', 'original', 'chunked', 'literalTranslation', 'naturalTranslation', 'grammarPoints'],
      },
    },
    vocabulary: {
      type: Type.ARRAY,
      description: '지문 또는 영작 문제에 포함된 수능·내신 핵심 어휘 및 숙어 4~10개',
      items: {
        type: Type.OBJECT,
        properties: {
          word: { type: Type.STRING, description: '원형 기준 영단어 또는 숙어' },
          partOfSpeech: { type: Type.STRING, description: '품사 약어 (n., v., adj., adv., prep., phr.)' },
          meaning: { type: Type.STRING, description: '문맥에 맞는 핵심 뜻 (한국어)' },
          synonyms: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '동의어 1~3개',
          },
          antonyms: {
            type: Type.ARRAY,
            items: { type: Type.STRING },
            description: '반의어 0~2개',
          },
          exampleInPassage: { type: Type.STRING, description: '지문 또는 모범 답안 내 활용 구절' },
        },
        required: ['word', 'partOfSpeech', 'meaning', 'synonyms', 'antonyms', 'exampleInPassage'],
      },
    },
  },
  required: [
    'title',
    'languageFormat',
    'detectedPostType',
    'extractedText',
    'koreanInstruction',
    'givenWords',
    'conditions',
    'summaryKorean',
    'isProblem',
    'questionType',
    'correctAnswer',
    'answerSummary',
    'evidenceSentences',
    'choices',
    'isWriting',
    'writingType',
    'writingKoreanPrompt',
    'writingContextTemplate',
    'writingModelAnswer',
    'writingAlternativeAnswers',
    'writingSteps',
    'writingDeductionPoints',
    'sentences',
    'vocabulary',
  ],
};

const gradeWritingSchema = {
  type: Type.OBJECT,
  properties: {
    verdict: {
      type: Type.STRING,
      description: '완벽히 맞으면 "correct", 일부 조건/어법이 아쉬우면 "partial", 의미·문법 오류가 크면 "needs_revision"',
    },
    scoreLabel: {
      type: Type.STRING,
      description: '한눈에 보이는 판정 요약 (예: "조건 2/2 충족 · 만점 수준", "어형 변화 1건 보완 필요")',
    },
    conditionChecks: {
      type: Type.ARRAY,
      description: '문제가 제시한 각 <조건> 및 문맥 요건에 대한 항목별 검증 결과',
      items: {
        type: Type.OBJECT,
        properties: {
          conditionText: { type: Type.STRING, description: '검증한 조건 또는 핵심 채점 기준' },
          passed: { type: Type.BOOLEAN, description: '충족 여부' },
          comment: { type: Type.STRING, description: '충족/미충족 사유에 대한 구체적 한국어 코멘트' },
        },
        required: ['conditionText', 'passed', 'comment'],
      },
    },
    correctedSentence: {
      type: Type.STRING,
      description: '학생 답안을 바탕으로 문법·어형·조건을 교정한 가장 자연스러운 완성 영어 문장',
    },
    feedbackKorean: {
      type: Type.STRING,
      description:
        '학생이 잘 쓴 점, 틀린 부분의 문법적 이유, 앞뒤 영어 문맥과의 연결 여부를 짚어주는 친절한 한국어 첨삭 총평 (2~5문장)',
    },
  },
  required: ['verdict', 'scoreLabel', 'conditionChecks', 'correctedSentence', 'feedbackKorean'],
};

const SYSTEM_INSTRUCTION = `당신은 대한민국 중·고등학교 내신 및 대학수학능력시험(수능) 영어 1타 강사이자 정밀한 영어 교육 에디터입니다.
사용자가 올린 사진 또는 텍스트는 다음 3가지 형태 중 하나입니다:
(A) **영어 문제 / 원문 (english)**: 영어 지문 + 영어 선지(①~⑤) 또는 순수 영어 독해 원문
(B) **한국어 문제 (korean)**: 우리말 제시문 + <보기> + <조건> 기반 영작 서술형 문제, 또는 한국어로 출제된 영어 문법·어휘 문제
(C) **한·영 혼합 문제 (mixed)**: 영어 본문 지문 + 한국어 발문·<조건>·<보기>가 결합된 서술형/요약문 빈칸 영작, 영어 문장 중간에 [우리말]이 들어간 부분 영작, 또는 영어 지문 + 한국어 선지(①~⑤) 객관식

반드시 지킬 원칙:
1. **언어 구성(languageFormat) 및 유형(detectedPostType) 정밀 판별**:
   - 영작/단어 배열/서술형 빈칸 완성이 핵심 과제이면 detectedPostType을 "writing", isWriting을 true로 설정하세요.
   - 객관식(①~⑤)이나 단답 선택 문제이면 detectedPostType을 "problem", isProblem을 true로 설정하세요. (영어 지문 + 한국어 선지 문제도 "problem"이며 languageFormat은 "mixed"입니다!)
   - 문제 없이 순수 영어 글 해석만 필요하면 detectedPostType을 "passage"로 설정하세요.
   - 주의: 한·영 혼합 서술형(영어 지문 + 한국어 영작 조건)인 경우에도 **영어 본문 전체의 sentences(직독직해·구문)와 evidenceSentences(영작 단서 문장)**를 빠짐없이 작성하세요!
2. **한국어 발문·<보기>·<조건> 분리 추출**:
   - koreanInstruction, givenWords, conditions에 문제 속 한국어 지시문, <보기> 단어, <조건>을 정확히 분리해 담으세요.
   - 영어 문장 속 일부 빈칸/괄호만 영작하는 문제라면 writingContextTemplate에 앞뒤 영어 문맥(예: "People believe that [ ________ ] is key.")을 넣으세요.
3. **영작 4단계 빌드업(writingSteps)**:
   - 영작 문제일 때 ① 핵심 뼈대(주어·동사) 잡기 → ② 수식어·연결어 확장 → ③ <조건> 및 어형 변형 적용 → ④ 최종 모범 답안 완성 순서로 4단계를 작성하고, writingDeductionPoints에 흔한 감점 요소를 담으세요.
4. **직독직해·구문·어휘 공통 생성**:
   - 영어 지문이 있는 문제는 본문 문장을, 한국어 단독 영작 문제는 완성된 모범 영어 답안 문장들을 sentences 배열에 넣어 "/" 끊어읽기 직독직해와 핵심 문법 포인트(문장당 1~2개 간결하게)를 제공하세요. vocabulary는 가장 중요한 4~8개 위주로 엄선하세요.
5. **추측 및 창작 절대 금지**:
   - 이미지가 백지이거나, 너무 흐리거나, 영어·한국어 문제 텍스트가 전혀 보이지 않으면 절대 임의의 지문이나 문제를 지어내지 마세요. 그 경우 반드시 extractedText를 빈 문자열("")로, sentences와 vocabulary를 빈 배열([])로 반환하고 summaryKorean에만 "사진에서 문제 텍스트를 읽을 수 없습니다."처럼 사유를 적으세요.
6. **정답·모범답안 이중 검증(100% 정확도 원칙)**:
   - 객관식 문제에서는 ①~⑤ 모든 선지를 지문 문장(evidenceSentences) 및 문법 규칙과 일대일로 대조·검산한 뒤 단 하나의 확실한 정답을 확정하세요. 어법/어휘/내용 일치·불일치 문제는 오답 4개가 왜 틀렸는지(혹은 맞는지) 명확한 본문 근거를 들어 설명하세요.
   - 영작·서술형 문제에서는 <보기> 단어 전부 사용 여부, 어형 변형, 시제·수일치·태, <조건>의 단어 수·필수 구문, 빈칸 문맥(writingContextTemplate)과의 문법적 연결까지 최종 검산한 뒤 writingModelAnswer를 확정하세요.
   - 사진이 여러 장(2장 이상) 첨부된 경우 모든 사진이 하나의 연속된 시험 지문·문제·조건이므로 순서를 이어 통합 분석하세요.`;

function safeParseModelJson(rawText: string): Record<string, unknown> | null {
  const cleaned = rawText
    .trim()
    .replace(/^```json\s*/i, '')
    .replace(/^```\s*/i, '')
    .replace(/```$/i, '')
    .trim();
  if (!cleaned) return null;
  try {
    return JSON.parse(cleaned) as Record<string, unknown>;
  } catch {
    const firstBrace = cleaned.indexOf('{');
    const lastBrace = cleaned.lastIndexOf('}');
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      try {
        return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1)) as Record<string, unknown>;
      } catch {
        return null;
      }
    }
    return null;
  }
}

function formatGeminiError(err: unknown): { status: number; message: string } {
  const rawMessage = err instanceof Error ? err.message : String(err);
  const lower = rawMessage.toLowerCase();
  if (rawMessage.includes('GEMINI_API_KEY')) {
    return { status: 500, message: rawMessage };
  }
  if (lower.includes('403') || lower.includes('permission_denied') || lower.includes('api_key_invalid')) {
    return {
      status: 403,
      message: 'Gemini API 키 인증에 실패했습니다. 환경 변수 GEMINI_API_KEY를 확인해 주세요.',
    };
  }
  if (lower.includes('429') || lower.includes('resource_exhausted')) {
    return {
      status: 429,
      message: '요청 한도를 잠시 초과했습니다. 10~20초 후 다시 시도해 주세요.',
    };
  }
  if (lower.includes('503') || lower.includes('unavailable') || lower.includes('high demand')) {
    return {
      status: 503,
      message: 'AI 분석 서버에 일시적으로 요청이 몰리고 있습니다. 잠시 후 다시 시도해 주세요.',
    };
  }
  if (lower.includes('504') || lower.includes('deadline') || lower.includes('timeout') || lower.includes('aborted')) {
    return {
      status: 504,
      message: '분석 시간이 초과되었습니다. 문제 영역을 더 좁게 자르거나 다시 시도해 주세요.',
    };
  }
  return {
    status: 500,
    message: `요청 처리 중 오류가 발생했습니다: ${rawMessage}`,
  };
}

function describeSolvePreference(
  solvePreference?: string,
  solveRequestNote?: string
): string {
  const map: Record<string, string> = {
    beginner:
      '초보자·기초 눈높이 풀이: 어려운 문법 용어는 쉬운 말로 풀어서 설명하고, 문장의 주어·동사·목적어 뼈대부터 친절하고 차근차근 설명하세요.',
    step_by_step:
      '단계별 논리 풀이: 정답이나 모범 답안으로 가는 과정을 1단계 → 2단계 → 3단계 순서로 끊어서 사고 흐름이 보이게 설명하세요.',
    concise:
      '핵심 요약·속전속결 풀이: 불필요한 서론 없이 핵심 근거 문장, 정답 이유, 오답 핵심 함정만 굵고 명확하게 짚어주세요.',
    exam_trap:
      '내신·수능 실전 변형 및 함정 대비 풀이: 출제자가 파놓은 오답 함정, 내신 서술형 변형 포인트, 어형 변화 주의점을 집중적으로 강조하세요.',
    balanced:
      '기본 균형 풀이: 정확한 정답 근거, 직독직해, 핵심 구문과 어휘를 균형 있게 제공하세요.',
  };
  const base = map[solvePreference || 'balanced'] || map.balanced;
  return solveRequestNote && solveRequestNote.trim()
    ? `${base}\n학생의 추가 풀이 방식 요청: "${solveRequestNote.trim()}" (이 요청을 반드시 최우선으로 반영해 해설을 작성하세요.)`
    : base;
}

function decodeHtmlEntities(str: string): string {
  return str
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/&nbsp;/g, ' ');
}

function cleanHtmlText(str: string): string {
  return decodeHtmlEntities(str.replace(/<[^>]+>/g, ' '))
    .replace(/새 창 열림/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

async function searchNaverWeb(
  query: string
): Promise<Array<{ title: string; url: string; domain: string; snippet: string }>> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 8000);
  try {
    const target = `https://search.naver.com/search.naver?where=nexearch&query=${encodeURIComponent(query)}`;
    const res = await fetch(target, {
      signal: controller.signal,
      headers: {
        'User-Agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept-Language': 'ko-KR,ko;q=0.9,en;q=0.8',
      },
    });
    if (!res.ok) return [];
    const html = await res.text();
    const byUrl = new Map<string, { title: string; url: string; domain: string; snippet: string }>();
    const anchorRe = /<a[^>]+href="(https?:\/\/[^"]+)"[^>]*>([\s\S]*?)<\/a>/gi;
    let match: RegExpExecArray | null;
    while ((match = anchorRe.exec(html)) !== null) {
      const href = decodeHtmlEntities(match[1]);
      if (
        /shopping\.naver|ader\.naver|nid\.naver|help\.naver|login|search\.naver|ssl\.pstatic|terms\.naver|kin\.naver\.com\/qna\/question/i.test(
          href
        )
      ) {
        continue;
      }
      let parsedUrl: URL;
      try {
        parsedUrl = new URL(href);
      } catch {
        continue;
      }
      const host = parsedUrl.hostname.replace(/^www\./, '');
      if (/^(?:m\.)?blog\.naver\.com$/i.test(host) && parsedUrl.pathname.split('/').filter(Boolean).length < 2) {
        continue;
      }
      if (parsedUrl.pathname.length <= 1 && !parsedUrl.search) continue;

      const text = cleanHtmlText(match[2]);
      if (
        text.length < 10 ||
        /^\d{4}\.\d{1,2}\.\d{1,2}\.?$/.test(text) ||
        /^\d+\s*(일|주|개월|년|시간|분)\s*전$/.test(text)
      ) {
        continue;
      }

      const prev = byUrl.get(href) || { title: '', url: href, domain: host, snippet: '' };
      if (
        !prev.title ||
        (text.length <= 110 && (prev.title.length > 110 || text.length > prev.title.length))
      ) {
        if (prev.title && prev.title.length > text.length && !prev.snippet) {
          prev.snippet = prev.title.slice(0, 320);
        }
        prev.title = text;
      } else if (text.length > prev.snippet.length && text !== prev.title) {
        prev.snippet = text.slice(0, 320);
      }
      byUrl.set(href, prev);
    }

    return Array.from(byUrl.values())
      .filter((item) => item.title.length >= 10 && !/^(https?:\/\/|www\.)/i.test(item.title))
      .slice(0, 8);
  } catch {
    return [];
  } finally {
    clearTimeout(timer);
  }
}

export function createApiApp() {
  const app = express();
  app.use(express.json({ limit: '30mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    res.json({ ok: true, model: PRIMARY_MODEL, fallbackModel: FALLBACK_MODEL });
  });

  app.get('/api/export-project-zip', (_req: Request, res: Response) => {
    try {
      const rootDir = resolveProjectRoot();
      const allowedRootFiles = [
        'package.json',
        'tsconfig.json',
        'vite.config.ts',
        'vercel.json',
        'index.html',
        'server.ts',
        'metadata.json',
        '.gitignore',
        '.env.example',
        'README.md',
      ];
      const allowedDirs = ['src', 'api'];

      const entries: Array<{ name: string; data: Buffer; mtime: Date }> = [];

      const pushSafeFile = (absPath: string, zipRelativePath: string) => {
        const resolved = path.resolve(absPath);
        if (!resolved.startsWith(rootDir + path.sep)) return;
        const stat = fs.lstatSync(resolved);
        if (stat.isSymbolicLink() || !stat.isFile()) return;
        const baseName = path.basename(resolved);
        if (baseName.startsWith('.env') && baseName !== '.env.example') return;
        entries.push({
          name: `passagenote-app/${zipRelativePath.replace(/\\/g, '/')}`,
          data: fs.readFileSync(resolved),
          mtime: stat.mtime,
        });
      };

      for (const fileName of allowedRootFiles) {
        const candidate = path.resolve(rootDir, fileName);
        if (fs.existsSync(candidate)) {
          pushSafeFile(candidate, fileName);
        }
      }

      const walkDir = (dirAbs: string, relBase: string) => {
        const resolvedDir = path.resolve(dirAbs);
        if (!resolvedDir.startsWith(rootDir + path.sep)) return;
        for (const item of fs.readdirSync(resolvedDir, { withFileTypes: true })) {
          if (item.isSymbolicLink()) continue;
          if (
            item.name === 'node_modules' ||
            item.name === 'dist' ||
            item.name === '.git' ||
            item.name === '.vercel'
          ) {
            continue;
          }
          const nextAbs = path.join(resolvedDir, item.name);
          const nextRel = path.posix.join(relBase, item.name);
          if (item.isDirectory()) {
            walkDir(nextAbs, nextRel);
          } else if (item.isFile()) {
            pushSafeFile(nextAbs, nextRel);
          }
        }
      };

      for (const dirName of allowedDirs) {
        const dirAbs = path.resolve(rootDir, dirName);
        if (fs.existsSync(dirAbs)) {
          walkDir(dirAbs, dirName);
        }
      }

      const hasReadme = entries.some((e) => e.name === 'passagenote-app/README.md');
      if (!hasReadme) {
        const readmeText = [
          '# PassageNote — 영어 원문·문제 사진 풀이 학습 노트',
          '',
          '수능·모의고사·내신 영어 객관식 문제, 한국어/한·영 혼합 서술형 영작, 원문 독해를 사진(최대 6장) 또는 텍스트로 분석하고 1:1 AI 튜터와 학습할 수 있는 웹 앱입니다.',
          '',
          '## Vercel 배포 방법 (권장)',
          '',
          '1. 이 폴더의 파일들을 GitHub 저장소에 업로드합니다.',
          '2. Vercel에서 해당 GitHub 저장소를 Import 합니다. (`vercel.json`과 `api/index.ts`가 자동으로 인식되며, npm `cache-dir` 충돌 방지 설정이 포함되어 있습니다.)',
          '3. **Project Settings → Environment Variables**에 `GEMINI_API_KEY` 값을 추가합니다.',
          '4. **Deploy**를 누르면 프론트엔드(Vite)와 백엔드 API(`/api/*`)가 함께 배포됩니다.',
          '5. 만약 이전 배포 캐시가 남아 오류가 난다면 Vercel **Deployments → Redeploy** 시 **Use existing Build Cache** 체크를 해제하고 재배포해 주세요.',
          '',
          '## 로컬 실행 방법',
          '',
          '```bash',
          'npm install',
          'cp .env.example .env',
          '# .env 파일에 GEMINI_API_KEY 값을 입력하세요',
          'npm run dev',
          '```',
          '',
          '## 프로덕션 빌드 및 실행',
          '',
          '```bash',
          'npm run build',
          'npm start',
          '```',
          '',
          '> `.env` 등 실제 API 키 비밀 파일은 보안을 위해 압축 파일에 포함되지 않습니다.',
          '',
        ].join('\n');
        entries.push({
          name: 'passagenote-app/README.md',
          data: Buffer.from(readmeText, 'utf8'),
          mtime: new Date(),
        });
      }

      const crcTable = (() => {
        const table = new Uint32Array(256);
        for (let n = 0; n < 256; n++) {
          let c = n;
          for (let k = 0; k < 8; k++) {
            c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
          }
          table[n] = c >>> 0;
        }
        return table;
      })();

      const crc32 = (buf: Buffer): number => {
        let c = 0xffffffff;
        for (let i = 0; i < buf.length; i++) {
          c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
        }
        return (c ^ 0xffffffff) >>> 0;
      };

      const toDosDateTime = (d: Date) => {
        const year = Math.max(1980, d.getFullYear());
        const dosTime =
          ((d.getHours() & 0x1f) << 11) |
          ((d.getMinutes() & 0x3f) << 5) |
          ((d.getSeconds() >> 1) & 0x1f);
        const dosDate =
          (((year - 1980) & 0x7f) << 9) |
          (((d.getMonth() + 1) & 0x0f) << 5) |
          (d.getDate() & 0x1f);
        return { dosTime, dosDate };
      };

      const localChunks: Buffer[] = [];
      const centralChunks: Buffer[] = [];
      let offset = 0;

      for (const entry of entries) {
        const nameBuf = Buffer.from(entry.name, 'utf8');
        const crc = crc32(entry.data);
        const compressed = zlib.deflateRawSync(entry.data);
        const { dosTime, dosDate } = toDosDateTime(entry.mtime);

        const localHeader = Buffer.alloc(30);
        localHeader.writeUInt32LE(0x04034b50, 0);
        localHeader.writeUInt16LE(20, 4);
        localHeader.writeUInt16LE(0x0800, 6);
        localHeader.writeUInt16LE(8, 8);
        localHeader.writeUInt16LE(dosTime, 10);
        localHeader.writeUInt16LE(dosDate, 12);
        localHeader.writeUInt32LE(crc, 14);
        localHeader.writeUInt32LE(compressed.length, 18);
        localHeader.writeUInt32LE(entry.data.length, 22);
        localHeader.writeUInt16LE(nameBuf.length, 26);
        localHeader.writeUInt16LE(0, 28);

        localChunks.push(localHeader, nameBuf, compressed);

        const centralHeader = Buffer.alloc(46);
        centralHeader.writeUInt32LE(0x02014b50, 0);
        centralHeader.writeUInt16LE(20, 4);
        centralHeader.writeUInt16LE(20, 6);
        centralHeader.writeUInt16LE(0x0800, 8);
        centralHeader.writeUInt16LE(8, 10);
        centralHeader.writeUInt16LE(dosTime, 12);
        centralHeader.writeUInt16LE(dosDate, 14);
        centralHeader.writeUInt32LE(crc, 16);
        centralHeader.writeUInt32LE(compressed.length, 20);
        centralHeader.writeUInt32LE(entry.data.length, 24);
        centralHeader.writeUInt16LE(nameBuf.length, 28);
        centralHeader.writeUInt16LE(0, 30);
        centralHeader.writeUInt16LE(0, 32);
        centralHeader.writeUInt16LE(0, 34);
        centralHeader.writeUInt16LE(0, 36);
        centralHeader.writeUInt32LE(0, 38);
        centralHeader.writeUInt32LE(offset, 42);

        centralChunks.push(centralHeader, nameBuf);
        offset += localHeader.length + nameBuf.length + compressed.length;
      }

      const centralDirSize = centralChunks.reduce((sum, b) => sum + b.length, 0);
      const eocd = Buffer.alloc(22);
      eocd.writeUInt32LE(0x06054b50, 0);
      eocd.writeUInt16LE(0, 4);
      eocd.writeUInt16LE(0, 6);
      eocd.writeUInt16LE(entries.length, 8);
      eocd.writeUInt16LE(entries.length, 10);
      eocd.writeUInt32LE(centralDirSize, 12);
      eocd.writeUInt32LE(offset, 16);
      eocd.writeUInt16LE(0, 20);

      const zipBuffer = Buffer.concat([...localChunks, ...centralChunks, eocd]);
      const dateStr = new Date().toISOString().slice(0, 10);

      res.setHeader('Content-Type', 'application/zip');
      res.setHeader(
        'Content-Disposition',
        `attachment; filename="passagenote-github-${dateStr}.zip"`
      );
      res.setHeader('Content-Length', String(zipBuffer.length));
      res.send(zipBuffer);
    } catch (err) {
      console.error('Error in /api/export-project-zip:', err);
      res.status(500).json({ error: '프로젝트 ZIP 파일을 생성하지 못했습니다.' });
    }
  });

  app.post('/api/analyze', async (req: Request, res: Response) => {
    try {
      const {
        imageBase64,
        mimeType,
        images,
        textOverride,
        requestedType = 'auto',
        sourceTag = '',
        solvePreference = 'balanced',
        solveRequestNote = '',
      } = req.body as {
        imageBase64?: string;
        mimeType?: string;
        images?: Array<{ imageBase64: string; mimeType?: string }>;
        textOverride?: string;
        requestedType?: 'auto' | 'problem' | 'writing' | 'passage';
        sourceTag?: string;
        solvePreference?: string;
        solveRequestNote?: string;
      };

      const imageList: Array<{ imageBase64: string; mimeType: string }> = [];
      if (Array.isArray(images) && images.length > 0) {
        for (const item of images.slice(0, 6)) {
          if (item && typeof item.imageBase64 === 'string' && item.imageBase64.trim()) {
            imageList.push({
              imageBase64: item.imageBase64,
              mimeType: item.mimeType || 'image/jpeg',
            });
          }
        }
      } else if (imageBase64 && imageBase64.trim()) {
        imageList.push({
          imageBase64,
          mimeType: mimeType || 'image/jpeg',
        });
      }

      if (imageList.length === 0 && (!textOverride || !textOverride.trim())) {
        res.status(400).json({ error: '분석할 이미지나 문제/원문 텍스트를 입력해 주세요.' });
        return;
      }

      const ai = getAiClient();

      const modeHint =
        requestedType === 'problem'
          ? '사용자가 이 자료를 [객관식·문제 풀이] 모드로 지정했습니다. 문제와 정답/오답 선지 해설을 충실히 작성하세요.'
          : requestedType === 'writing'
            ? '사용자가 이 자료를 [영작·서술형] 모드로 지정했습니다. <조건>과 <보기>를 분석하고 4단계 영작 빌드업, 모범 답안, 감점 포인트를 충실히 작성하세요.'
            : requestedType === 'passage'
              ? '사용자가 이 자료를 [순수 원문 독해] 모드로 지정했습니다. 문장별 직독직해, 구문 분석, 어휘 정리에 집중하세요.'
              : '자료가 영어 문제/원문인지, 한국어 영작 문제인지, 한·영 혼합 문제인지 스스로 판별하여 알맞은 섹션을 빠짐없이 채우세요.';

      const sourceHint = sourceTag.trim() ? `사용자가 입력한 출처/메모: "${sourceTag.trim()}"` : '';
      const styleHint = `[원하는 풀이·해설 방식]\n${describeSolvePreference(solvePreference, solveRequestNote)}`;
      const multiImageHint =
        imageList.length > 1
          ? `총 ${imageList.length}장의 사진이 순서대로 첨부되었습니다. 각 사진의 본문, 문제 발문, <보기>, <조건>, ①~⑤ 선택지를 빠짐없이 연결하여 하나의 완전한 학습 노트로 통합 분석하세요.`
          : '';

      const promptText = [
        modeHint,
        sourceHint,
        styleHint,
        multiImageHint,
        textOverride && textOverride.trim()
          ? `다음 영어/한국어 또는 한·영 혼합 문제(원문)를 분석해 주세요:\n\n${textOverride.trim()}`
          : '첨부된 사진 속 영어 원문, 영어 문제, 한국어 문제, 또는 한·영 혼합 문제를 정밀 판독하고 검증된 정답과 학습 노트로 분석해 주세요.',
      ]
        .filter(Boolean)
        .join('\n\n');

      const parts: Array<{ inlineData?: { mimeType: string; data: string }; text?: string }> = [];

      imageList.forEach((imgItem, idx) => {
        if (imageList.length > 1) {
          parts.push({ text: `[사진 ${idx + 1}]` });
        }
        const cleanedBase64 = imgItem.imageBase64.includes(',')
          ? imgItem.imageBase64.split(',')[1]
          : imgItem.imageBase64;
        parts.push({
          inlineData: {
            mimeType: imgItem.mimeType,
            data: cleanedBase64,
          },
        });
      });
      parts.push({ text: promptText });

      const response = await generateWithFallback(
        ai,
        { parts },
        {
          systemInstruction: SYSTEM_INSTRUCTION,
          responseMimeType: 'application/json',
          responseSchema: analysisSchema,
        },
        24000,
        ThinkingLevel.MEDIUM
      );

      const rawText = (response.text ?? '').trim();
      if (!rawText) {
        res.status(502).json({ error: '모델로부터 분석 결과를 받지 못했습니다. 다시 시도해 주세요.' });
        return;
      }

      const parsed = safeParseModelJson(rawText) as Record<string, any> | null;
      if (!parsed) {
        console.error('Malformed JSON from /api/analyze, length:', rawText.length);
        res.status(502).json({
          error: '지문이 너무 길거나 사진이 복잡해 분석 응답이 중간에 끊겼습니다. 문단 단위로 영역을 잘라 다시 시도해 주세요.',
        });
        return;
      }

      if (!parsed.extractedText || !String(parsed.extractedText).trim()) {
        res.status(422).json({
          error:
            parsed.summaryKorean ||
            '이미지에서 문제나 원문 텍스트를 인식하지 못했습니다. 더 선명하게 자르거나 텍스트를 직접 입력해 주세요.',
        });
        return;
      }

      const langFormat =
        parsed.languageFormat === 'korean' || parsed.languageFormat === 'mixed'
          ? parsed.languageFormat
          : 'english';

      let detectedType: 'problem' | 'writing' | 'passage' = 'passage';
      if (parsed.detectedPostType === 'writing' || (parsed.isWriting && !parsed.isProblem)) {
        detectedType = 'writing';
      } else if (parsed.detectedPostType === 'problem' || parsed.isProblem) {
        detectedType = 'problem';
      }

      const normalized = {
        title: parsed.title || '영어 학습 노트',
        languageFormat: langFormat,
        detectedPostType: detectedType,
        extractedText: parsed.extractedText,
        koreanInstruction: parsed.koreanInstruction || '',
        givenWords: Array.isArray(parsed.givenWords) ? parsed.givenWords.map(String) : [],
        conditions: Array.isArray(parsed.conditions) ? parsed.conditions.map(String) : [],
        summaryKorean: parsed.summaryKorean || '',
        problemSection:
          parsed.isProblem || detectedType === 'problem'
            ? {
                questionType: parsed.questionType || '문제 풀이',
                correctAnswer: parsed.correctAnswer || '',
                answerSummary: parsed.answerSummary || '',
                evidenceSentences: Array.isArray(parsed.evidenceSentences) ? parsed.evidenceSentences : [],
                choices: Array.isArray(parsed.choices) ? parsed.choices : [],
              }
            : null,
        writingSection:
          parsed.isWriting || detectedType === 'writing'
            ? {
                writingType: parsed.writingType || '서술형 영작',
                koreanPrompt: parsed.writingKoreanPrompt || parsed.koreanInstruction || '',
                contextTemplate: parsed.writingContextTemplate || '',
                givenWords: Array.isArray(parsed.givenWords) ? parsed.givenWords.map(String) : [],
                conditions: Array.isArray(parsed.conditions) ? parsed.conditions.map(String) : [],
                evidenceSentences: Array.isArray(parsed.evidenceSentences) ? parsed.evidenceSentences : [],
                modelAnswer: parsed.writingModelAnswer || '',
                alternativeAnswers: Array.isArray(parsed.writingAlternativeAnswers)
                  ? parsed.writingAlternativeAnswers.map(String)
                  : [],
                steps: Array.isArray(parsed.writingSteps) ? parsed.writingSteps : [],
                deductionPoints: Array.isArray(parsed.writingDeductionPoints)
                  ? parsed.writingDeductionPoints.map(String)
                  : [],
              }
            : null,
        sentences: Array.isArray(parsed.sentences) ? parsed.sentences : [],
        vocabulary: Array.isArray(parsed.vocabulary)
          ? parsed.vocabulary.map((v: Record<string, unknown>, idx: number) => ({
              id: `vocab-${Date.now()}-${idx}`,
              word: String(v.word || ''),
              partOfSpeech: String(v.partOfSpeech || ''),
              meaning: String(v.meaning || ''),
              synonyms: Array.isArray(v.synonyms) ? v.synonyms.map(String) : [],
              antonyms: Array.isArray(v.antonyms) ? v.antonyms.map(String) : [],
              exampleInPassage: String(v.exampleInPassage || ''),
              memorized: false,
            }))
          : [],
      };

      res.json(normalized);
    } catch (err) {
      console.error('Error in /api/analyze:', err);
      const { status, message } = formatGeminiError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/grade-writing', async (req: Request, res: Response) => {
    try {
      const {
        extractedText,
        koreanPrompt,
        contextTemplate,
        givenWords = [],
        conditions = [],
        modelAnswer,
        userAnswer,
      } = req.body as {
        extractedText?: string;
        koreanPrompt?: string;
        contextTemplate?: string;
        givenWords?: string[];
        conditions?: string[];
        modelAnswer?: string;
        userAnswer?: string;
      };

      if (!userAnswer || !userAnswer.trim()) {
        res.status(400).json({ error: '채점할 영작 답안을 입력해 주세요.' });
        return;
      }

      const ai = getAiClient();

      const gradingPrompt = [
        extractedText ? `[전체 문제/지문 원문]\n${extractedText}` : '',
        koreanPrompt ? `[우리말 제시문/요구사항]\n${koreanPrompt}` : '',
        contextTemplate ? `[앞뒤 영어 문맥 틀 (빈칸 위치 확인)]\n${contextTemplate}` : '',
        givenWords.length > 0 ? `[<보기> 주어진 단어]\n${givenWords.join(', ')}` : '',
        conditions.length > 0
          ? `[<조건> 필수 준수 사항]\n${conditions.map((c, i) => `${i + 1}. ${c}`).join('\n')}`
          : '[<조건>]\n자연스럽고 문법적으로 정확한 영어 문장으로 완성할 것',
        modelAnswer ? `[참고 모범 답안]\n${modelAnswer}` : '',
        `[학생이 제출한 영작 답안]\n${userAnswer.trim()}`,
      ]
        .filter(Boolean)
        .join('\n\n');

      const response = await generateWithFallback(ai, gradingPrompt, {
        systemInstruction:
          '당신은 대한민국 중·고등학교 내신 영어 서술형 채점관입니다. 학생이 작성한 영작 답안을 <조건>, <보기> 단어 사용, 어형 변화, 문법·철자, 그리고 한·영 혼합 문맥 틀(contextTemplate)과의 호응까지 꼼꼼하게 채점하세요. 조건이 없다면 기본 문법·의미 전달·자연스러움을 기준으로 2~3개 체크 항목을 만들어 검증하세요. 모범 답안과 표현이 조금 달라도 문법과 조건을 모두 만족하면 정답("correct")으로 인정하세요.',
        responseMimeType: 'application/json',
        responseSchema: gradeWritingSchema,
      });

      const rawText = (response.text ?? '').trim();
      if (!rawText) {
        res.status(502).json({ error: '첨삭 결과를 받지 못했습니다. 다시 시도해 주세요.' });
        return;
      }

      const parsed = safeParseModelJson(rawText) as Record<string, any> | null;
      if (!parsed) {
        res.status(502).json({ error: '첨삭 응답 형식을 해석하지 못했습니다. 다시 시도해 주세요.' });
        return;
      }
      const verdict =
        parsed.verdict === 'correct' || parsed.verdict === 'partial'
          ? parsed.verdict
          : 'needs_revision';

      res.json({
        verdict,
        scoreLabel: parsed.scoreLabel || (verdict === 'correct' ? '정답' : '보완 필요'),
        conditionChecks: Array.isArray(parsed.conditionChecks) ? parsed.conditionChecks : [],
        correctedSentence: parsed.correctedSentence || modelAnswer || '',
        feedbackKorean: parsed.feedbackKorean || '',
      });
    } catch (err) {
      console.error('Error in /api/grade-writing:', err);
      const { status, message } = formatGeminiError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/followup', async (req: Request, res: Response) => {
    try {
      const {
        extractedText,
        summaryKorean,
        question,
        modelAnswer,
        solvePreference = 'balanced',
        history = [],
      } = req.body as {
        extractedText?: string;
        summaryKorean?: string;
        question?: string;
        modelAnswer?: string;
        solvePreference?: string;
        history?: Array<{ question: string; answer: string }>;
      };

      if (!question || !question.trim() || !extractedText) {
        res.status(400).json({ error: '질문 내용과 원문 컨텍스트가 필요합니다.' });
        return;
      }

      const ai = getAiClient();

      const historyText =
        history.length > 0
          ? history
              .slice(-4)
              .map((h) => `학생 질문: ${h.question}\n선생님 답변: ${h.answer}`)
              .join('\n\n')
          : '';

      const prompt = [
        `[학습 중인 원문/문제]\n${extractedText}`,
        summaryKorean ? `[핵심 요약]\n${summaryKorean}` : '',
        modelAnswer ? `[모범 영작 답안]\n${modelAnswer}` : '',
        `[학생이 요청한 풀이·설명 방식]\n${describeSolvePreference(solvePreference)}`,
        historyText ? `[이전 질의응답]\n${historyText}` : '',
        `[학생의 현재 질문]\n${question.trim()}`,
      ]
        .filter(Boolean)
        .join('\n\n');

      const response = await generateWithFallback(
        ai,
        prompt,
        {
          systemInstruction:
            '당신은 대한민국 최고 수준의 중·고등 영어 선생님입니다. 학생이 보고 있는 영어 지문, 한국어 영작 문제, 또는 한·영 혼합 문제의 정확한 원문 근거를 바탕으로 질문에 틀림없이 답변하세요. 먼저 핵심 결론(정답/핵심 이유)을 명확히 제시하고, 학생이 선택한 풀이 방식에 맞춰 이해하기 쉽게 설명해 주세요. 불필요한 마크다운 기호 없이 깔끔한 텍스트와 줄바꿈으로 작성하세요.',
        },
        18000,
        ThinkingLevel.MEDIUM
      );

      const answer = (response.text ?? '').trim();
      if (!answer) {
        res.status(502).json({ error: '답변을 생성하지 못했습니다.' });
        return;
      }

      res.json({ answer });
    } catch (err) {
      console.error('Error in /api/followup:', err);
      const { status, message } = formatGeminiError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/find-similar', async (req: Request, res: Response) => {
    try {
      const {
        title = '',
        extractedText = '',
        koreanInstruction = '',
        languageFormat = 'english',
        postType = 'passage',
        questionType = '',
        grammarPoints = [],
        focusMode = 'csat_variant',
        customQuery = '',
      } = req.body as {
        title?: string;
        extractedText?: string;
        koreanInstruction?: string;
        languageFormat?: 'english' | 'korean' | 'mixed';
        postType?: 'problem' | 'writing' | 'passage';
        questionType?: string;
        grammarPoints?: string[];
        focusMode?: 'csat_variant' | 'passage_origin' | 'grammar_writing';
        customQuery?: string;
      };

      if (!extractedText.trim() && !customQuery.trim()) {
        res.status(400).json({ error: '검색할 원문이나 문제 정보가 필요합니다.' });
        return;
      }

      const ai = getAiClient();

      const focusInstruction =
        focusMode === 'passage_origin'
          ? '이 영어 지문이나 문장의 원문 출처(모의고사 연도·월·번호, EBS 교재, 교과서 단원, 원서/기사)와 해당 지문 해설 웹페이지를 집중적으로 검색하세요.'
          : focusMode === 'grammar_writing'
            ? '이 문제에 쓰인 핵심 문법 구문, 어형 변화, 서술형 영작 패턴과 동일한 유형의 기출·연습 문제 및 설명 자료를 인터넷에서 검색하세요.'
            : '이 지문 또는 문제와 같은 모의고사·수능·내신 변형 문제, 서술형 예상 문제, 유사 주제 기출 문제를 인터넷에서 검색하세요.';

      const snippetText = extractedText.trim().slice(0, 1400);

      try {
        const searchPrompt = [
          `[검색 목적]\n${focusInstruction}`,
          title ? `[노트 제목]\n${title}` : '',
          `[문제 형태] ${languageFormat} / ${postType}${questionType ? ` / ${questionType}` : ''}`,
          koreanInstruction ? `[한국어 발문·조건]\n${koreanInstruction}` : '',
          grammarPoints.length > 0 ? `[핵심 구문·어법]\n${grammarPoints.slice(0, 6).join(', ')}` : '',
          customQuery.trim() ? `[학생이 추가로 입력한 검색어]\n${customQuery.trim()}` : '',
          snippetText ? `[문제·지문 발췌]\n${snippetText}` : '',
          `반드시 Google 검색 도구를 실행하여 실제 웹에서 관련 출처와 유사 문제 자료를 찾은 뒤, 순수 JSON 객체로만 정리해 주세요.`,
          `{
  "overviewKorean": "검색을 통해 확인한 출처 추정 및 유사 문제 핵심 요약 (2~4문장)",
  "items": [
    {
      "title": "검색된 유사 문제·기출·해설 자료 제목 (3~5건)",
      "sourceName": "출처 또는 시험/교재/사이트명",
      "matchReason": "현재 문제와 연결되는 지문·주제·문법·영작 조건",
      "problemSnippet": "검색된 문제 유형, 핵심 문장, 또는 변형 출제 포인트 요약",
      "studyTip": "대비 핵심 포인트",
      "url": "참고 URL (없으면 빈 문자열)"
    }
  ]
}`,
        ]
          .filter(Boolean)
          .join('\n\n');

        const groundedResp = await ai.models.generateContent({
          model: PRIMARY_MODEL,
          contents: searchPrompt,
          config: {
            tools: [{ googleSearch: {} }],
            thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
            httpOptions: { timeout: 14000 },
            systemInstruction:
              '당신은 대한민국 중고등 영어 내신·수능 자료 검색 전문가입니다. googleSearch 도구로 실제 웹 정보를 검색한 결과만 바탕으로 답변하고, 확인되지 않은 URL을 지어내지 마세요.',
          },
        });

        const rawText = (groundedResp.text ?? '').trim();
        const groundingMeta = groundedResp.candidates?.[0]?.groundingMetadata as
          | {
              webSearchQueries?: string[];
              groundingChunks?: Array<{ web?: { uri?: string; title?: string } }>;
            }
          | undefined;

        const webSourcesMap = new Map<string, { title: string; uri: string; domain: string }>();
        for (const chunk of groundingMeta?.groundingChunks || []) {
          const uri = chunk.web?.uri?.trim();
          if (!uri || !/^https?:\/\//i.test(uri)) continue;
          let domain = '';
          try {
            domain = new URL(uri).hostname.replace(/^www\./, '');
          } catch {
            domain = '';
          }
          if (!webSourcesMap.has(uri)) {
            webSourcesMap.set(uri, {
              title: chunk.web?.title?.trim() || domain || '웹 출처',
              uri,
              domain,
            });
          }
        }

        const searchQueries = Array.isArray(groundingMeta?.webSearchQueries)
          ? groundingMeta.webSearchQueries.map(String).filter(Boolean)
          : [];
        const parsed = rawText ? (safeParseModelJson(rawText) as Record<string, any> | null) : null;
        const items =
          parsed && Array.isArray(parsed.items)
            ? parsed.items.slice(0, 6).map((item: Record<string, unknown>) => ({
                title: String(item.title || '유사 학습 자료'),
                sourceName: String(item.sourceName || '웹 검색 결과'),
                matchReason: String(item.matchReason || ''),
                problemSnippet: String(item.problemSnippet || ''),
                studyTip: String(item.studyTip || ''),
                url:
                  typeof item.url === 'string' && /^https?:\/\//i.test(item.url.trim())
                    ? item.url.trim()
                    : undefined,
              }))
            : [];

        if (webSourcesMap.size > 0 || items.length > 0) {
          res.json({
            searchedAt: Date.now(),
            focusMode,
            overviewKorean:
              (parsed && typeof parsed.overviewKorean === 'string' && parsed.overviewKorean.trim()) ||
              rawText ||
              '웹에서 관련 학습 자료를 검색했습니다.',
            searchQueries,
            items,
            webSources: Array.from(webSourcesMap.values()),
          });
          return;
        }
      } catch (groundingErr) {
        console.warn(
          'Google Search grounding skipped, using live server web search fallback:',
          groundingErr instanceof Error ? groundingErr.message : String(groundingErr)
        );
      }

      const queryGenPrompt = [
        `학생이 공부 중인 영어/한국어 문제에 대해 한국 웹(네이버 등)에서 비슷한 기출·내신 변형 문제와 해설을 찾기 위한 검색어 2개를 JSON으로 제안하세요.`,
        `검색 목적: ${focusInstruction}`,
        title ? `제목: ${title}` : '',
        grammarPoints.length > 0 ? `핵심 문법: ${grammarPoints.slice(0, 4).join(', ')}` : '',
        customQuery.trim() ? `학생 지정 키워드: ${customQuery.trim()}` : '',
        snippetText ? `지문/문제 발췌: ${snippetText.slice(0, 600)}` : '',
        `규칙: query1은 한국어 수능·내신·모의고사 검색용(예: "영어 내신 변형 문제 ..."), query2는 원문 영어 구문 5~8단어나 핵심 영작 구문 포함 검색용으로 짧고 정확하게 작성하세요.`,
      ]
        .filter(Boolean)
        .join('\n');

      const queryResp = await generateWithFallback(ai, queryGenPrompt, {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            query1: { type: Type.STRING },
            query2: { type: Type.STRING },
          },
          required: ['query1', 'query2'],
        },
      });

      const qParsed = safeParseModelJson(queryResp.text ?? '') as { query1?: string; query2?: string } | null;
      const queries = Array.from(
        new Set(
          [
            customQuery.trim(),
            qParsed?.query1?.trim() || `${title} 영어 변형 문제`,
            qParsed?.query2?.trim() || `${grammarPoints[0] || title} 영어 서술형 기출`,
          ].filter(Boolean)
        )
      ).slice(0, 3);

      const searchBatches = await Promise.all(queries.map((q) => searchNaverWeb(q)));
      const mergedByUrl = new Map<string, { title: string; url: string; domain: string; snippet: string }>();
      for (const batch of searchBatches) {
        for (const item of batch) {
          if (!mergedByUrl.has(item.url)) {
            mergedByUrl.set(item.url, item);
          }
        }
      }
      const retrievedResults = Array.from(mergedByUrl.values()).slice(0, 10);

      const synthesisPrompt = [
        `[학생의 현재 문제]`,
        title ? `제목: ${title}` : '',
        `형태: ${languageFormat} / ${postType}${questionType ? ` / ${questionType}` : ''}`,
        grammarPoints.length > 0 ? `핵심 구문: ${grammarPoints.slice(0, 5).join(', ')}` : '',
        snippetText ? `원문 발췌: ${snippetText.slice(0, 800)}` : '',
        `\n[인터넷 검색으로 실제 수집한 웹페이지 목록]`,
        retrievedResults.length > 0
          ? retrievedResults
              .map(
                (r, idx) =>
                  `[${idx + 1}] 제목: ${r.title}\n도메인: ${r.domain}\nURL: ${r.url}\n요약: ${r.snippet || '(본문 미리보기 없음)'}`
              )
              .join('\n\n')
          : '(직접 웹 결과 없음 — 이 문제의 핵심 문법·유형 기반 유사 변형 대비 가이드를 제시하고 url은 빈 문자열로 두세요.)',
        `\n위 실제 검색 결과만을 근거로, 학생의 문제와 관련성이 높은 자료 3~5건을 골라 JSON으로 정리하세요. 목록에 없는 URL은 절대 만들어내지 마세요.`,
      ]
        .filter(Boolean)
        .join('\n');

      const synthesisResp = await generateWithFallback(ai, synthesisPrompt, {
        responseMimeType: 'application/json',
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            overviewKorean: {
              type: Type.STRING,
              description: '인터넷 검색 결과 요약 및 이 문제의 예상 출제·변형 방향 (2~3문장)',
            },
            items: {
              type: Type.ARRAY,
              items: {
                type: Type.OBJECT,
                properties: {
                  title: { type: Type.STRING },
                  sourceName: { type: Type.STRING },
                  matchReason: { type: Type.STRING },
                  problemSnippet: { type: Type.STRING },
                  studyTip: { type: Type.STRING },
                  url: { type: Type.STRING },
                },
                required: ['title', 'sourceName', 'matchReason', 'problemSnippet', 'studyTip', 'url'],
              },
            },
          },
          required: ['overviewKorean', 'items'],
        },
        systemInstruction:
          '당신은 대한민국 중고등 영어 내신·수능 자료 큐레이터입니다. 제공된 실제 검색 결과(URL, 제목, 요약)를 학생 문제와 연결하여 가장 도움이 되는 유사 문제·해설 자료를 선별해 설명하세요. URL은 제공된 목록에 있는 것만 사용하세요.',
      });

      const synthParsed = safeParseModelJson(synthesisResp.text ?? '') as Record<string, any> | null;
      const allowedUrls = new Set(retrievedResults.map((r) => r.url));
      const items =
        synthParsed && Array.isArray(synthParsed.items)
          ? synthParsed.items.slice(0, 5).map((item: Record<string, unknown>) => {
              const rawUrl = typeof item.url === 'string' ? item.url.trim() : '';
              return {
                title: String(item.title || '유사 학습 자료'),
                sourceName: String(item.sourceName || '웹 검색 결과'),
                matchReason: String(item.matchReason || ''),
                problemSnippet: String(item.problemSnippet || ''),
                studyTip: String(item.studyTip || ''),
                url: rawUrl && allowedUrls.has(rawUrl) ? rawUrl : undefined,
              };
            })
          : [];

      const webSources = retrievedResults.map((r) => ({
        title: r.title,
        uri: r.url,
        domain: r.domain,
      }));

      res.json({
        searchedAt: Date.now(),
        focusMode,
        overviewKorean:
          (synthParsed && typeof synthParsed.overviewKorean === 'string' && synthParsed.overviewKorean.trim()) ||
          '인터넷 검색을 통해 관련 내신·수능 학습 자료를 찾았습니다. 아래 자료와 링크를 확인해 보세요.',
        searchQueries: queries,
        items,
        webSources,
      });
    } catch (err) {
      console.error('Error in /api/find-similar:', err);
      const { status, message } = formatGeminiError(err);
      res.status(status).json({ error: message });
    }
  });

  app.post('/api/chat', async (req: Request, res: Response) => {
    try {
      const {
        message,
        history = [],
        roleId = 'tutor',
        solvePreference = 'balanced',
        customCondition = '',
        customContent = '',
        context,
      } = req.body as {
        message?: string;
        history?: Array<{ role: 'user' | 'model'; text: string }>;
        roleId?: 'tutor' | 'writing_coach' | 'quick_quiz';
        solvePreference?: string;
        customCondition?: string;
        customContent?: string;
        context?: {
          title?: string;
          languageFormat?: string;
          postType?: string;
          extractedText?: string;
          summaryKorean?: string;
          modelAnswer?: string;
        } | null;
      };

      const trimmedMessage = typeof message === 'string' ? message.trim() : '';
      const trimmedCondition = typeof customCondition === 'string' ? customCondition.trim() : '';
      const trimmedContent = typeof customContent === 'string' ? customContent.trim() : '';

      if (!trimmedMessage && !trimmedContent) {
        res.status(400).json({ error: '질문 내용이나 참고할 지문·문장을 입력해 주세요.' });
        return;
      }

      const ai = getAiClient();

      const roleInstructions: Record<'tutor' | 'writing_coach' | 'quick_quiz', string> = {
        tutor:
          '당신은 대한민국 중·고등학교 내신 및 수능 영어 1타 튜터입니다. 영어 지문·한국어 문제·한·영 혼합 문제 모두에 대해 학생의 질문 맥락을 기억하며 문장 구조, 직독직해, 정확한 정답/오답 논리를 근거 문장과 함께 틀림없이 설명하세요. 학생이 어떤 방식으로 풀이해 달라고 요청하거나 풀이 스타일을 물어보면 즉시 선택지를 제안하고 그 방식대로 맞추어 설명하세요.',
        writing_coach:
          '당신은 내신 영어 서술형·영작 전담 코치입니다. 학생이 작성하려는 영어 문장이나 어순, 시제, 수일치, <조건> 적용을 꼼꼼히 검증하여 감점 위험이 없는 정확한 모범 문장과 단계별 힌트를 제시하세요.',
        quick_quiz:
          '당신은 스피드 영어 어휘·어법 퀴즈 파트너입니다. 학생이 요청하면 현재 지문 또는 중·고등 핵심 영문법·어휘를 바탕으로 1문항씩 빠르고 간결한 미니 퀴즈를 내고, 학생이 답하면 즉시 정·오답 판정과 한 줄 핵심 해설을 제공하세요.',
      };

      const styleBlock = `\n[선호 풀이 방식]\n${describeSolvePreference(solvePreference)}`;

      const conditionBlock = trimmedCondition
        ? `\n\n[학생이 직접 지정한 필수 답변 조건]\n${trimmedCondition.slice(0, 1200)}\n(위 조건과 형식·제약을 반드시 최우선으로 준수하여 답변하세요.)`
        : '';

      const contextBlock =
        context && context.extractedText
          ? [
              '\n[현재 학생이 펼쳐 둔 학습 노트 정보]',
              context.title ? `제목: ${context.title}` : '',
              context.languageFormat ? `형식: ${context.languageFormat} (${context.postType || ''})` : '',
              context.summaryKorean ? `핵심 요약: ${context.summaryKorean}` : '',
              context.modelAnswer ? `모범 영작 답안: ${context.modelAnswer}` : '',
              `원문 발췌:\n${context.extractedText.slice(0, 1800)}`,
            ]
              .filter(Boolean)
              .join('\n')
          : '';

      const systemInstruction = `${roleInstructions[roleId] || roleInstructions.tutor}${styleBlock}${conditionBlock}${contextBlock}\n\n답변은 과도한 마크다운 기호 없이 읽기 쉬운 문단과 줄바꿈으로 작성하고, 정답·문법 판단은 반드시 원문 근거를 대조해 확실하게 제시하세요.`;

      const formattedHistory = history
        .filter((item) => item && typeof item.text === 'string' && item.text.trim())
        .slice(-16)
        .map((item) => ({
          role: item.role === 'model' ? ('model' as const) : ('user' as const),
          parts: [{ text: item.text.trim() }],
        }));

      const effectiveUserPrompt = [
        trimmedCondition ? `[요청 조건]\n${trimmedCondition}` : '',
        trimmedContent ? `[질문할 내용/지문/문장]\n${trimmedContent.slice(0, 4000)}` : '',
        trimmedMessage
          ? `[질문]\n${trimmedMessage}`
          : '[질문]\n위 조건과 첨부한 내용에 맞춰 자세히 풀이·설명해 주세요.',
      ]
        .filter(Boolean)
        .join('\n\n');

      const preferredModel = roleId === 'quick_quiz' ? FALLBACK_MODEL : PRIMARY_MODEL;

      const sendWithModel = async (modelName: string, timeoutMs: number, level: ThinkingLevel) => {
        const chat = ai.chats.create({
          model: modelName,
          history: formattedHistory,
          config: {
            systemInstruction,
            thinkingConfig: { thinkingLevel: level },
            httpOptions: { timeout: timeoutMs },
          },
        });
        return await chat.sendMessage({ message: effectiveUserPrompt });
      };

      let replyText = '';
      let usedModel = preferredModel;

      try {
        const response = await sendWithModel(
          preferredModel,
          preferredModel === FALLBACK_MODEL ? 25000 : 18000,
          preferredModel === FALLBACK_MODEL ? ThinkingLevel.LOW : ThinkingLevel.MEDIUM
        );
        replyText = (response.text ?? '').trim();
      } catch (err) {
        if (preferredModel !== FALLBACK_MODEL && isTransientUpstreamError(err)) {
          console.warn('Primary chat model unavailable, falling back to:', FALLBACK_MODEL);
          usedModel = FALLBACK_MODEL;
          const fallbackResp = await sendWithModel(FALLBACK_MODEL, 30000, ThinkingLevel.LOW);
          replyText = (fallbackResp.text ?? '').trim();
        } else {
          throw err;
        }
      }

      if (!replyText) {
        res.status(502).json({ error: '튜터 답변을 생성하지 못했습니다. 다시 시도해 주세요.' });
        return;
      }

      res.json({
        reply: replyText,
        modelUsed: usedModel,
      });
    } catch (err) {
      console.error('Error in /api/chat:', err);
      const { status, message } = formatGeminiError(err);
      res.status(status).json({ error: message });
    }
  });

  return app;
}
