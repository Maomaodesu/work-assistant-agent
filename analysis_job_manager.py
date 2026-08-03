"""工作项自动发现的单进程后台任务管理器。"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

from external_conversation_sync import ExternalConversationSync
from project_matcher import ConversationProjectMatcher
from semantic_segmenter import SemanticConversationSegmenter
from work_item_discovery import WorkItemDiscovery, work_item_discovery
from workspace_store import WorkspaceStore, WorkspaceStoreError, workspace_store


ACTIVE_STATUSES = {"queued", "running", "paused"}


class AnalysisControl:
    def __init__(self):
        self._resume_event = threading.Event()
        self._resume_event.set()
        self._cancel_event = threading.Event()

    def pause(self):
        self._resume_event.clear()

    def resume(self):
        self._resume_event.set()

    def cancel(self):
        self._cancel_event.set()
        self._resume_event.set()

    def checkpoint(self) -> bool:
        while not self._resume_event.wait(0.2):
            if self._cancel_event.is_set():
                return False
        return not self._cancel_event.is_set()


@dataclass
class AnalysisJob:
    run_id: str
    request: dict
    control: AnalysisControl
    future: Future | None = None


class AnalysisJobManager:
    def __init__(
        self,
        store: WorkspaceStore | None = None,
        discovery: WorkItemDiscovery | None = None,
        conversation_sync: ExternalConversationSync | None = None,
        project_matcher: ConversationProjectMatcher | None = None,
        segmenter: SemanticConversationSegmenter | None = None,
        *,
        recover_interrupted: bool = True,
    ):
        self.store = store or workspace_store
        self.discovery = discovery or work_item_discovery
        # 每个管理器都绑定同一个 store，避免测试或多工作区任务误写入全局数据库。
        self.conversation_sync = conversation_sync or ExternalConversationSync(self.store)
        self.project_matcher = project_matcher or ConversationProjectMatcher(self.store)
        self.segmenter = segmenter or SemanticConversationSegmenter(self.store)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="work-analysis")
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()
        if recover_interrupted:
            self._recover_interrupted_runs()

    def _recover_interrupted_runs(self):
        for run in self.store.list_classification_runs(limit=200):
            if run["status"] in ACTIVE_STATUSES:
                self.store.update_classification_run(
                    run["run_id"], status="failed", stage="interrupted",
                    error_message="服务重启导致后台分析中断，可以点击重试继续增量分析。",
                )

    def start(
        self,
        *,
        project_id: str | None = None,
        force: bool = False,
        limit: int | None = None,
        retry_of_run_id: str | None = None,
    ) -> dict:
        if project_id and not self.store.get_project(project_id):
            raise WorkspaceStoreError("项目不存在")
        request = {"project_id": project_id, "force": bool(force), "limit": limit}
        with self._lock:
            duplicate = next(
                (
                    job for job in self._jobs.values()
                    if job.request.get("project_id") == project_id
                    and (job.future is None or not job.future.done())
                ),
                None,
            )
        if duplicate:
            raise WorkspaceStoreError(f"该项目已有分析任务正在运行：{duplicate.run_id}")
        # 先展示当前已知片段数；同步、匹配、分段完成后 discover 会用最新值覆盖。
        segments = self.store.list_segments(project_id)
        if limit is not None:
            segments = segments[:max(0, int(limit))]
        run = self.store.create_classification_run(
            run_type="full" if force else "incremental",
            total_sources=len(segments),
            request=request,
            retry_of_run_id=retry_of_run_id,
        )
        control = AnalysisControl()
        job = AnalysisJob(run_id=run["run_id"], request=request, control=control)
        with self._lock:
            self._jobs[run["run_id"]] = job
            job.future = self._executor.submit(self._execute, job)
        return self.store.get_classification_run(run["run_id"])

    def _execute(self, job: AnalysisJob):
        try:
            self.store.update_classification_run(
                job.run_id, status="running", stage="syncing_local_conversations"
            )
            if not job.control.checkpoint():
                self._mark_cancelled(job.run_id)
                return
            self.conversation_sync.sync()

            self.store.update_classification_run(job.run_id, stage="matching_projects")
            if not job.control.checkpoint():
                self._mark_cancelled(job.run_id)
                return
            self.project_matcher.match_all()

            self.store.update_classification_run(job.run_id, stage="segmenting_conversations")
            if not job.control.checkpoint():
                self._mark_cancelled(job.run_id)
                return
            self.segmenter.segment_all()

            if not job.control.checkpoint():
                self._mark_cancelled(job.run_id)
                return
            self.discovery.discover(
                project_id=job.request.get("project_id"),
                force=bool(job.request.get("force")),
                limit=job.request.get("limit"),
                run_id=job.run_id,
                control=job.control,
            )
        except Exception as exc:
            run = self.store.get_classification_run(job.run_id)
            if run and run["status"] not in {"completed", "cancelled", "failed"}:
                self.store.update_classification_run(
                    job.run_id, status="failed", stage="failed",
                    error_message=str(exc),
                )
        finally:
            with self._lock:
                self._jobs.pop(job.run_id, None)

    def _mark_cancelled(self, run_id: str):
        """在预处理阶段响应取消；发现器阶段仍由发现器自身完成取消收尾。"""
        run = self.store.get_classification_run(run_id)
        if run and run["status"] not in {"completed", "failed", "cancelled"}:
            self.store.update_classification_run(
                run_id, status="cancelled", stage="cancelled",
                error_message="用户取消了分析任务",
            )

    def _job(self, run_id: str) -> AnalysisJob | None:
        with self._lock:
            return self._jobs.get(run_id)

    def pause(self, run_id: str) -> dict:
        run = self.store.get_classification_run(run_id)
        if not run:
            raise WorkspaceStoreError("分析任务不存在")
        job = self._job(run_id)
        if not job or run["status"] not in {"queued", "running"}:
            raise WorkspaceStoreError("当前分析任务无法暂停")
        job.control.pause()
        return self.store.update_classification_run(run_id, status="paused")

    def resume(self, run_id: str) -> dict:
        run = self.store.get_classification_run(run_id)
        if not run:
            raise WorkspaceStoreError("分析任务不存在")
        job = self._job(run_id)
        if not job or run["status"] != "paused":
            raise WorkspaceStoreError("当前分析任务无法继续")
        job.control.resume()
        return self.store.update_classification_run(run_id, status="running")

    def cancel(self, run_id: str) -> dict:
        run = self.store.get_classification_run(run_id)
        if not run:
            raise WorkspaceStoreError("分析任务不存在")
        if run["status"] not in ACTIVE_STATUSES:
            raise WorkspaceStoreError("当前分析任务无法取消")
        job = self._job(run_id)
        if job:
            job.control.cancel()
        return self.store.update_classification_run(
            run_id, status="cancelled", stage="cancellation_requested",
            error_message="用户取消了分析任务",
        )

    def retry(self, run_id: str) -> dict:
        previous = self.store.get_classification_run(run_id)
        if not previous:
            raise WorkspaceStoreError("分析任务不存在")
        if previous["status"] not in {"failed", "cancelled"}:
            raise WorkspaceStoreError("只有失败或已取消的任务可以重试")
        request = previous.get("request") or {}
        return self.start(
            project_id=request.get("project_id"),
            force=False,
            limit=request.get("limit"),
            retry_of_run_id=run_id,
        )

    def get(self, run_id: str) -> dict | None:
        return self.store.get_classification_run(run_id)

    def list(self, limit: int = 20) -> list[dict]:
        return self.store.list_classification_runs(limit=limit)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


analysis_job_manager = AnalysisJobManager()
