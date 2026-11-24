import { useCallback, useEffect, useMemo, useState } from 'react';
import { Camera, Loader2, MessageSquareText } from 'lucide-react';
import { AppView, PostType, StudyPost, TutorChatContext } from './types/study';
import {
  bulkSavePosts,
  clearAllPosts,
  deletePost,
  initializeAndLoadPosts,
  restoreSamplePosts,
  savePost,
} from './lib/storage';
import { fileToDataUrl } from './lib/imageUtils';
import { TopBar } from './components/TopBar';
import { FeedSidebar } from './components/FeedSidebar';
import { StudyWorkspace } from './components/StudyWorkspace';
import { SnapStudioModal } from './components/SnapStudioModal';
import { VocabVaultView } from './components/VocabVaultView';
import { BackupView } from './components/BackupView';
import { TutorChatDrawer } from './components/TutorChatDrawer';

export default function App() {
  const [posts, setPosts] = useState<StudyPost[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [activeView, setActiveView] = useState<AppView>('feed');
  const [selectedPostId, setSelectedPostId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [typeFilter, setTypeFilter] = useState<'all' | PostType>('all');

  const [snapModalOpen, setSnapModalOpen] = useState<boolean>(false);
  const [pastedImageUrl, setPastedImageUrl] = useState<string | null>(null);
  const [tutorOpen, setTutorOpen] = useState<boolean>(false);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const loaded = await initializeAndLoadPosts();
        if (!mounted) return;
        setPosts(loaded);
        if (loaded.length > 0) {
          setSelectedPostId(loaded[0].id);
        }
      } catch (err) {
        if (!mounted) return;
        setLoadError(
          err instanceof Error ? err.message : '로컬 저장소를 초기화하지 못했습니다.'
        );
      } finally {
        if (mounted) setLoading(false);
      }
    })();
    return () => {
      mounted = false;
    };
  }, []);

  // Global Ctrl+V / Cmd+V paste support for screenshots of English passages
  useEffect(() => {
    const onPaste = async (e: ClipboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTypingField =
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.isContentEditable);

      const items = e.clipboardData?.items;
      if (!items) return;

      for (const item of Array.from(items)) {
        if (item.type.startsWith('image/')) {
          if (isTypingField && !snapModalOpen) continue;
          const file = item.getAsFile();
          if (!file) continue;
          e.preventDefault();
          try {
            const url = await fileToDataUrl(file);
            setPastedImageUrl(url);
            setSnapModalOpen(true);
          } catch {
            // ignore unreadable clipboard image
          }
          break;
        }
      }
    };

    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [snapModalOpen]);

  const filteredPosts = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    return posts.filter((post) => {
      if (activeView === 'review' && post.studyStatus !== 'review_needed') {
        return false;
      }
      if (typeFilter !== 'all' && post.postType !== typeFilter) {
        return false;
      }
      if (!q) return true;
      return (
        post.title.toLowerCase().includes(q) ||
        post.sourceTag.toLowerCase().includes(q) ||
        post.analysis.extractedText.toLowerCase().includes(q) ||
        (post.analysis.koreanInstruction || '').toLowerCase().includes(q) ||
        (post.analysis.writingSection?.modelAnswer || '').toLowerCase().includes(q) ||
        post.analysis.vocabulary.some((v) => v.word.toLowerCase().includes(q))
      );
    });
  }, [posts, activeView, typeFilter, searchQuery]);

  const selectedPost = useMemo(() => {
    const pool = filteredPosts.length > 0 ? filteredPosts : posts;
    return pool.find((p) => p.id === selectedPostId) || filteredPosts[0] || null;
  }, [filteredPosts, posts, selectedPostId]);

  const activeTutorContext: TutorChatContext | null = useMemo(() => {
    if (!selectedPost) return null;
    return {
      postId: selectedPost.id,
      title: selectedPost.title,
      languageFormat: selectedPost.languageFormat,
      postType: selectedPost.postType,
      extractedText: selectedPost.analysis.extractedText,
      summaryKorean: selectedPost.analysis.summaryKorean,
      modelAnswer: selectedPost.analysis.writingSection?.modelAnswer || undefined,
    };
  }, [selectedPost]);

  const reviewCount = useMemo(
    () => posts.filter((p) => p.studyStatus === 'review_needed').length,
    [posts]
  );

  const vocabUnmemorizedCount = useMemo(
    () =>
      posts.reduce(
        (sum, p) => sum + p.analysis.vocabulary.filter((v) => !v.memorized).length,
        0
      ),
    [posts]
  );

  const handleCreatedPost = useCallback(async (newPost: StudyPost) => {
    await savePost(newPost);
    setPosts((prev) => [newPost, ...prev]);
    setSelectedPostId(newPost.id);
    setActiveView('feed');
    setPastedImageUrl(null);
  }, []);

  const handleUpdatePost = useCallback(async (updated: StudyPost) => {
    await savePost(updated);
    setPosts((prev) => prev.map((p) => (p.id === updated.id ? updated : p)));
  }, []);

  const handleDeletePost = useCallback(async (id: string) => {
    await deletePost(id);
    setPosts((prev) => {
      const remaining = prev.filter((p) => p.id !== id);
      setSelectedPostId((curr) => (curr === id ? remaining[0]?.id || null : curr));
      return remaining;
    });
  }, []);

  const handleToggleVocabFromVault = useCallback(
    async (postId: string, vocabId: string) => {
      const target = posts.find((p) => p.id === postId);
      if (!target) return;
      const updated: StudyPost = {
        ...target,
        updatedAt: Date.now(),
        analysis: {
          ...target.analysis,
          vocabulary: target.analysis.vocabulary.map((v) =>
            v.id === vocabId ? { ...v, memorized: !v.memorized } : v
          ),
        },
      };
      await handleUpdatePost(updated);
    },
    [posts, handleUpdatePost]
  );

  const handleImportPosts = useCallback(async (imported: StudyPost[]): Promise<number> => {
    await bulkSavePosts(imported);
    const mergedMap = new Map<string, StudyPost>();
    for (const p of posts) mergedMap.set(p.id, p);
    for (const p of imported) mergedMap.set(p.id, p);
    const nextList = Array.from(mergedMap.values()).sort((a, b) => b.createdAt - a.createdAt);
    setPosts(nextList);
    if (!selectedPostId && nextList[0]) {
      setSelectedPostId(nextList[0].id);
    }
    return imported.length;
  }, [posts, selectedPostId]);

  const handleRestoreSamples = useCallback(async () => {
    const refreshed = await restoreSamplePosts();
    setPosts(refreshed);
    if (refreshed[0]) setSelectedPostId(refreshed[0].id);
  }, []);

  const handleClearAll = useCallback(async () => {
    await clearAllPosts();
    setPosts([]);
    setSelectedPostId(null);
  }, []);

  const openSnapStudio = () => {
    setPastedImageUrl(null);
    setSnapModalOpen(true);
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      <TopBar
        activeView={activeView}
        onSelectView={(v) => setActiveView(v)}
        onOpenSnapStudio={openSnapStudio}
        reviewCount={reviewCount}
        vocabUnmemorizedCount={vocabUnmemorizedCount}
      />

      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {loading ? (
          <div className="py-24 flex flex-col items-center justify-center text-slate-500 gap-3">
            <Loader2 className="w-6 h-6 animate-spin text-blue-700" aria-hidden="true" />
            <p className="text-sm">학습 노트를 불러오는 중입니다...</p>
          </div>
        ) : loadError ? (
          <div className="max-w-lg mx-auto my-12 p-6 bg-white border border-red-200 rounded-xl text-center space-y-2">
            <p className="text-sm font-semibold text-red-800">저장소를 열 수 없습니다</p>
            <p className="text-xs text-red-700">{loadError}</p>
          </div>
        ) : activeView === 'vocab' ? (
          <VocabVaultView
            posts={posts}
            onToggleVocabMemorized={handleToggleVocabFromVault}
            onOpenPost={(postId) => {
              setSelectedPostId(postId);
              setActiveView('feed');
            }}
          />
        ) : activeView === 'backup' ? (
          <BackupView
            posts={posts}
            onImportPosts={handleImportPosts}
            onRestoreSamples={handleRestoreSamples}
            onClearAll={handleClearAll}
          />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            <div className="lg:col-span-4 xl:col-span-3">
              <FeedSidebar
                posts={filteredPosts}
                selectedId={selectedPost?.id || null}
                onSelectPost={(id) => setSelectedPostId(id)}
                searchQuery={searchQuery}
                onSearchChange={setSearchQuery}
                typeFilter={typeFilter}
                onTypeFilterChange={setTypeFilter}
                onlyReviewMode={activeView === 'review'}
              />
            </div>

            <div className="lg:col-span-8 xl:col-span-9">
              {selectedPost ? (
                <StudyWorkspace
                  post={selectedPost}
                  onUpdatePost={handleUpdatePost}
                  onDeletePost={handleDeletePost}
                  onOpenSnapStudio={openSnapStudio}
                  onOpenTutor={() => setTutorOpen(true)}
                />
              ) : (
                <div className="bg-white border border-slate-200 rounded-xl p-10 sm:p-16 text-center space-y-4">
                  <div className="w-12 h-12 rounded-xl bg-blue-50 text-blue-700 flex items-center justify-center mx-auto">
                    <Camera className="w-6 h-6" aria-hidden="true" />
                  </div>
                  <div className="space-y-1.5 max-w-md mx-auto">
                    <h2 className="font-editorial text-xl font-bold text-slate-900">
                      {activeView === 'review'
                        ? '복습이 필요한 문제가 없습니다'
                        : '첫 영어 원문이나 문제를 찍어 올려보세요'}
                    </h2>
                    <p className="text-sm text-slate-600 leading-relaxed">
                      {activeView === 'review'
                        ? '모든 노트를 이해 완료로 정리하셨습니다. 새 문제를 찍어 학습 피드를 채워보세요.'
                        : '사진을 촬영·업로드하거나 스크린샷을 Ctrl+V로 붙여넣으면 직독직해, 구문 분석, 핵심 어휘, 정답 및 오답 해설이 자동으로 생성됩니다.'}
                    </p>
                  </div>
                  <div>
                    <button
                      type="button"
                      onClick={openSnapStudio}
                      className="inline-flex items-center gap-2 px-5 py-2.5 text-sm font-semibold text-white bg-blue-700 hover:bg-blue-800 rounded-lg cursor-pointer"
                    >
                      <Camera className="w-4 h-4" aria-hidden="true" />
                      <span>새 문제 찍기</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <SnapStudioModal
        isOpen={snapModalOpen}
        initialImageDataUrl={pastedImageUrl}
        onClose={() => {
          setSnapModalOpen(false);
          setPastedImageUrl(null);
        }}
        onCreatedPost={handleCreatedPost}
      />

      {!tutorOpen && (
        <button
          type="button"
          onClick={() => setTutorOpen(true)}
          className="fixed bottom-5 right-5 z-40 inline-flex items-center gap-2 px-4 py-3 text-xs sm:text-sm font-semibold text-white bg-slate-900 hover:bg-slate-800 border border-slate-700 rounded-xl shadow-lg transition-all cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-700"
        >
          <MessageSquareText className="w-4 h-4 text-blue-400 shrink-0" aria-hidden="true" />
          <span>AI 튜터 채팅</span>
        </button>
      )}

      <TutorChatDrawer
        isOpen={tutorOpen}
        onClose={() => setTutorOpen(false)}
        activeContext={activeTutorContext}
      />
    </div>
  );
}
