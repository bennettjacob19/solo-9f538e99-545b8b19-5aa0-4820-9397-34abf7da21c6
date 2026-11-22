"""强停语义单测（纯 stdlib，模拟后端 + 真实子进程）：

  python3 -m unittest tests.test_stop -v

覆盖：
  - 启动中的强停：等本轮 spawn 出结果、确认 SIGKILL 发出后才答复
  - 信号发出即给回执，oom_kill 计数随后才读取
  - 进程已退出（含“已退出未收尾”窗口）或启动失败：不发信号、不计入归因
  - 一个槽位的强停在等待时，另一槽位照常接单、强停
"""
from __future__ import annotations

import asyncio
import unittest

from app.cgroups import SimulatedBackend
from app.jobs import JobManager


class BrokenSpawnBackend(SimulatedBackend):
    """child_argv 指向不存在的可执行文件，复现“启动失败”。"""

    def child_argv(self, handle, inner):
        return ["/nonexistent/gsb-definitely-missing"]


class StopTestBase(unittest.IsolatedAsyncioTestCase):
    backend_factory = SimulatedBackend

    async def asyncSetUp(self):
        self.backend = self.backend_factory(slots=2, limit_mib=256, reserve_mib=96)
        self.mgr = JobManager(self.backend)

    async def asyncTearDown(self):
        await self.mgr.shutdown()
        self.backend.cleanup()

    async def wait_finished(self, job, timeout=15):
        for _ in range(int(timeout / 0.05)):
            if job.state == "finished":
                return
            await asyncio.sleep(0.05)
        self.fail(f"job {job.id} 未及时完成")

    async def wait_reap_window(self, job, timeout=15):
        """等到“子进程已退出但归因未落定”的窗口（OOM_SETTLE_SECONDS 内）。"""
        for _ in range(int(timeout / 0.01)):
            proc = self.mgr._procs.get(job.id)
            if proc is not None and proc.returncode is not None:
                return proc
            if job.state == "finished":
                return None
            await asyncio.sleep(0.01)
        self.fail(f"job {job.id} 子进程迟迟不退出")


class StopDuringStartupTest(StopTestBase):
    async def test_stop_during_startup_waits_for_spawn_then_sends(self):
        job = await self.mgr.start(16, 30)
        # 不让出事件循环、直接强停：此时 spawn 尚未完成，stop 必须等待
        outcome = await self.mgr.stop(job.id)
        self.assertTrue(outcome.sent, outcome.detail)
        self.assertTrue(job.stop_requested)          # 计入归因
        self.assertIsNotNone(job.pid)                # 回执时本轮子进程已存在
        self.assertEqual(job.state, "running")       # 回执不等作业收尾
        self.assertIsNone(job.oom_kill_after)        # 计数随后才读取
        await self.wait_finished(job)
        self.assertEqual(job.exit_signal, 9)
        self.assertEqual(job.verdict, "force_stopped")

    async def test_other_slot_serves_while_stop_waits(self):
        a = await self.mgr.start(16, 30)
        t = asyncio.create_task(self.mgr.stop(a.id))  # 启动中，挂起等 spawn
        b = await self.mgr.start(16, 30)              # 另一槽照常接单
        ob = await self.mgr.stop(b.id)                # 另一槽照常强停
        oa = await t
        self.assertTrue(oa.sent, oa.detail)
        self.assertTrue(ob.sent, ob.detail)
        self.assertNotEqual(a.slot, b.slot)
        await self.wait_finished(a)
        await self.wait_finished(b)
        self.assertEqual(a.verdict, "force_stopped")
        self.assertEqual(b.verdict, "force_stopped")


class StopAfterExitTest(StopTestBase):
    async def test_stop_after_finish_not_counted(self):
        job = await self.mgr.start(4, 0.2)
        await self.wait_finished(job)
        self.assertEqual(job.verdict, "normal_exit")
        outcome = await self.mgr.stop(job.id)
        self.assertFalse(outcome.sent)
        self.assertFalse(job.stop_requested)         # 归因忽略本次请求
        self.assertEqual(job.verdict, "normal_exit")

    async def test_stop_in_reap_window_not_counted(self):
        """子进程已退出、只差读完计数：强停必须拒发信号，归因不受污染。"""
        job = await self.mgr.start(4, 0.2)
        proc = await self.wait_reap_window(job)
        if proc is not None and job.state == "running":
            # 逮到窗口：进程已死但 state 仍 running —— 原 bug 场景
            outcome = await self.mgr.stop(job.id)
            self.assertFalse(outcome.sent, outcome.detail)
            self.assertFalse(job.stop_requested)
        await self.wait_finished(job)
        self.assertEqual(job.verdict, "normal_exit")
        self.assertFalse(job.stop_requested)

    async def test_stop_after_oom_exit_keeps_oom_verdict(self):
        """原 bug：OOM 已杀死子进程，窗口内强停曾把死因污染成 unknown。"""
        slot_max_mib = self.backend.handles[0].memory_max_bytes // (1024 * 1024)
        job = await self.mgr.start(slot_max_mib * 2, 60)  # 明确超限 -> OOM
        proc = await self.wait_reap_window(job)
        if proc is not None and job.state == "running":
            outcome = await self.mgr.stop(job.id)
            self.assertFalse(outcome.sent, outcome.detail)
        await self.wait_finished(job)
        self.assertEqual(job.exit_signal, 9)
        self.assertGreaterEqual(job.oom_kill_delta or 0, 1)
        self.assertFalse(job.stop_requested)
        self.assertEqual(job.verdict, "oom_kill")    # 死因不再变成 unknown


class StopSpawnFailureTest(StopTestBase):
    backend_factory = BrokenSpawnBackend

    async def test_stop_after_spawn_failure_not_counted(self):
        job = await self.mgr.start(16, 30)
        outcome = await self.mgr.stop(job.id)  # 启动中发出，等 spawn 失败结果
        self.assertFalse(outcome.sent, outcome.detail)
        self.assertTrue(job.spawn_failed)
        self.assertFalse(job.stop_requested)
        await self.wait_finished(job)
        self.assertFalse(job.stop_requested)


class StopUnknownJobTest(StopTestBase):
    async def test_unknown_job(self):
        outcome = await self.mgr.stop("no-such-job")
        self.assertIsNone(outcome.job)
        self.assertFalse(outcome.sent)


if __name__ == "__main__":
    unittest.main()
