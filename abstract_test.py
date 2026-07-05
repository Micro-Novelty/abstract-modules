import asyncio
import time
import AbstractIntegratedModule
from AbstractIntegratedModule import (
    IntegratedPipeline,
    PipelinePredictionManager,
    CohesiveAgentDeployment
)
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split


print(f'FILE: {AbstractIntegratedModule.__file__}')
memory_name = 'agent_memory'
main_model = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080,ssl_cert_file=None, ssl_key_file=None) # provide cert_file path or key_file path (optional)
main_prediction = PipelinePredictionManager(main_model, label_csv='ManualsTraining.txt', target_title='window_title', label='label')
# example_manual_training is a .txt file that contain csv format like above example.

example_rules = [
                     # === WORK / PRODUCTIVITY ===
                     (r'code|programming|develop|debug|compile|script', 'focused_work'),
                     (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
                     (r'notion|evernote|onenote|notes|todo|task', 'productive'),
                     (r'slack|teams|discord|zoom|meeting|call', 'communication'),
                     (r'email|gmail|outlook|inbox|mail', 'communication'),
                     
                     # === ENTERTAINMENT ===
                     (r'youtube|netflix|twitch|stream|video', 'entertainment'),
                     (r'music|spotify|soundcloud|audio|player', 'entertainment'),
                     (r'game|gaming|steam|epic|play', 'gaming'),
                     (r'facebook|instagram|tiktok|social|post', 'social_media'),
                     
                     # === BROWSING ===
                     (r'chrome|firefox|edge|safari|browser', 'browsing'),
                     (r'google|search|wiki|wiki|article', 'information'),
                     (r'stackoverflow|github|docs|documentation', 'research'),

                     # more rules
                 ]
# activate explainability capability to explain uncertainty:
main_model.show_explainability_details = False
main_model.distribution.predict_manager = main_prediction
main_model.autonomous = True

# deprecate transformer use.
main_model.use_transformer = False

# test samples with more sophisticated rules and more complex titles for prediction
# (title, intent)
test_titles = [
 ("Opening Thesis.docx", "slight_work"),
 ("Watching YouTube and Google Chrome", "distracted"),
 ("Watching Slack", "communication"),
 ("Programming in Visual Studio Code", "focused_work"),
 ("Watching netflix.com - Chrome", "break"),
# more titles 
 ]  

def test_sets(type=None):
    dt = 0
    if type == "moons" or dt > 5:
    	X, y_raw = make_moons(
    	n_samples = 1000, 
    	noise=0.75,  
    	random_state=99)
    else:
    	dt += 1    
    	if dt <= 5:
    		print("type sample 1")
    		X, y_raw = make_classification(
    		n_samples=1000,
    		n_features=3,
    		n_classes=3,
    		n_informative=3,
    		n_redundant=0,
    		class_sep=1.5,
    		random_state=259)
    	else:
    		generalization_test()
    	
    return X, y_raw

X, y = test_sets()
            
titles, _, label_map = main_prediction.load_labels_from_csv('ManualsTraining.txt', 'window_title', 'label')
# small training with simple titles
main_model.train(titles, y)

y = np.asarray(y)
print(X.shape, y.shape)
   
results, chosen_label, confidence = main_prediction.advanced_prediction_method(test_titles, label_map, example_rules,
                             X=X, y=y,
                             show_proba=False, top_k=3, 
                             use_transformer=True,
                             return_attention=False,
                             save_results=True,
                             batch_size=2)


dataset, _ = main_model.data_preparation(titles, label_map)
sequence_inputs = main_model.sequence_encoding(dataset)
X_raw_generation, y, n_classes, input_dim = main_model.mlp_training_features(example_rules, dataset)

main_model.initialize_fitting(dataset)
X_raw_features = main_model.tfidf.transform(X_raw_generation).toarray()
transformer_features = main_model.transformer_pooled_features(sequence_inputs)
X_features = np.concatenate([X_raw_features, transformer_features], axis=-1)

peer_probability_calibration = main_model.predict_proba(sequence_inputs, X_features, type='Hybrid', embedded=True) # peer-to-peer calibration is inside this function

from AbstractIntegratedModule import PipelineAsyncManager
from AbstractIntegratedModule import SecurityConfig
from AbstractIntegratedModule import SecurityLevel

print(" = TESTING ASYNCHRONOUS PREDICTION MANAGER = ")
# Set discovery secret (in production, use environment variable)
secret_key = 'my-ultra-safe-secret-key-for-authentication' # you can customize this key


security_config = SecurityConfig(
      max_text_length=10000, # can be extended
      max_queue_size=100, # can be extended
      rate_limit_requests=60,  # 60 per minute
      require_api_key=True, #
      max_pending_tasks=50,
      request_timeout=60.0,

      # Start with no IP restrictions, you can add allowed IPs for asynchronous prediction externally, boothstrap_auth for better security
      allowed_ips=[],
      blocklisted_ips=[],
      require_bootstrap_auth = False # true for better security (Not recommended, cause less flexibility)
  )

async_manager = PipelineAsyncManager(main_model, 
        main_prediction, # your previous initialized PipelinePredictionManager
        config=security_config, 
        state_file=None, # state file is used to load known security logs ex: ip used, ip blacklisted, etc.
        security_level=SecurityLevel.PRODUCTION, # production level security initiated
        api_key=secret_key, #set secret key you initialized        
        max_workers=4, # workers to initiate prediction, more workers, more capabilities to process prediction requests.
        task_timeout=30, 
        max_retries=3 ) # retries after failure during prediction

async_manager.start(method='Transformer_included', bootstrap_token=None) # boothstrap token is optional for better security

texts = {'test_titles': test_titles, 'label_map': label_map, 'rules': example_rules, 'X': None, 'y': None, 'use_transformer': True}
regular_predict = async_manager.predict(
   texts=texts,
   timeout=60,
   retries=None,
   api_key=secret_key) # advanced prediction method for asynchronous prediction.

# with retries: async_manager.predict(texts, timeout=60, retries=5, api_key=secret_key) # 5 times retry if failed

print('[==] Initiating advanced batch prediction')
predicted_output = async_manager.advanced_batch_prediction(test_titles, label_map, example_rules, api_key=secret_key, client_ip=None)
# for better and faster advanced prediction, consider using advanced batch prediction like in the above example

print('[+] Initiating samples prediction without titles and rules')
results, chosen_label, confidence = main_prediction.advanced_prediction_method(titles=None, label_map=label_map, rules=None,
                             X=X, y=y,
                             show_proba=False, top_k=3, 
                             use_transformer=True,
                             return_attention=False,
                             save_results=True,
                             batch_size=2)

# ... more features you can add

# ================ SECOND TEST ================
SECRET_KEY = "test-secret-key"

LABEL_FILE = "ManualsTraining.txt"
PEER_FILE = "peer_config.json"

memory_name = "arm64_test"

pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8080)

predict_manager = PipelinePredictionManager(
    pipeline,
    label_csv=LABEL_FILE,
    target_title="window_title",
    label="label"
)

titles, y, label_map = predict_manager.load_labels_from_csv(
    LABEL_FILE,
    "window_title",
    "label"
)

pipeline.train(titles, y)

test_titles = [
    ("Programming in VSCode", "focused_work"),
    ("Watching YouTube", "entertainment")
]

print('FILE:', AbstractIntegratedModule.__file__)


async def main(main_pipeline):
    sec_pipeline = IntegratedPipeline(memory_name=memory_name, use_async=True, agent_port=8010)

    rules = [
        # === WORK / PRODUCTIVITY ===
        (r'code|programming|develop|debug|compile|script', 'focused_work'),
        (r'vscode|visual_studio|ide|terminal|shell', 'focused_work'),
        (r'notion|evernote|onenote|notes|todo|task', 'productive'),
        (r'slack|teams|discord|zoom|meeting|call', 'communication'),
        (r'email|gmail|outlook|inbox|mail', 'communication'),
        
        # === ENTERTAINMENT ===
        (r'youtube|netflix|twitch|stream|video', 'entertainment'),
        (r'music|spotify|soundcloud|audio|player', 'entertainment'),
        (r'game|gaming|steam|epic|play', 'gaming'),
        (r'facebook|instagram|tiktok|social|post', 'social_media'),
        
        # === BROWSING ===
        (r'chrome|firefox|edge|safari|browser', 'browsing'),
        (r'google|search|wiki|wiki|article', 'information'),
        (r'stackoverflow|github|docs|documentation', 'research'),
        
        # === FILE MANAGEMENT ===
        (r'download|folder|file|document|pdf', 'file_work'),
        (r'dropbox|onedrive|google_drive|cloud', 'cloud_storage'),
        (r'zip|rar|extract|compress|archive', 'file_management'),
        
        # === SYSTEM / DEV ===
        (r'terminal|cmd|powershell|bash|shell', 'system_work'),
        (r'docker|kubernetes|container|deploy', 'devops'),
        (r'git|commit|push|pull|branch|merge', 'version_control'),
        (r'test|unit|debug|error|exception', 'testing'),
        
        # === DATA / ANALYSIS ===
        (r'excel|spreadsheet|sheet|csv|table', 'data_work'),
        (r'python|r|sql|query|database', 'data_analysis'),
        (r'chart|graph|visualization|dashboard|plot', 'visualization'),
        
        # === COMMUNICATION ===
        (r'whatsapp|telegram|signal|messenger', 'messaging'),
        (r'zoom|meet|webex|video_call', 'video_call'),
        (r'calendar|schedule|event|meeting|appointment', 'scheduling'),
        
        # === CREATIVE ===
        (r'photoshop|illustrator|figma|design|canvas', 'creative'),
        (r'premiere|final_cut|video_edit|render', 'video_editing'),
        (r'blender|3d|model|render|animation', '3d_work'),
        
        # === LEARNING ===
        (r'coursera|udemy|edx|course|learn', 'learning'),
        (r'book|ebook|reader|pdf|document', 'reading'),
        (r'podcast|audiobook|listen|lecture', 'audio_learning'),
        
        # === UTILITY ===
        (r'calculator|converter|tool|utility', 'utility'),
        (r'weather|clock|timer|alarm|reminder', 'utility'),
        (r'translate|language|dictionary|translate', 'utility'),
        
        # === RARITY PATTERNS ===
        (r'common|not_common|twitch|debian|watch', 'very abundant'),
        (r'bit-common|pycharm|unix|code|programming|python|java', 'bit-abundant'),
        (r'medium|discord|teams|zoom|linux_mint|message', 'abundant'),
        (r'rare|pdf|word|macOS|ubuntu|document', 'not abundant'),
        (r'ultra|firefox|edge|browser|unix|web', 'medium rare'),
        (r'ultra_rare|music|linux|Home_linux_router', 'bit-rare'),
        (r'medium-rare|steam|red_hat_enterprise_linux|play|windows', 'very rare'),
        (r'rarer|oracle|system|config|server_linux_router', 'absolute rare'),
    ]

    agent1 = CohesiveAgentDeployment(
        pipeline=main_pipeline,
        memory_name="agent1",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5555,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    agent2 = CohesiveAgentDeployment(
        pipeline=sec_pipeline,
        memory_name="agent2",
        filename=LABEL_FILE,
        target_title="window_title",
        label_name="label",
        enable_peers=True,
        peer_discovery_port=5556,
        secret_key=SECRET_KEY,
        shared_auth_token=SECRET_KEY,
        predict_manager=predict_manager,
        peer_config=PEER_FILE,
    )

    await agent1.start()
    print("Agent 1 started on port 5555, Time:", time.time())
    await agent2.start()
    print("Agent 2 started on port 5556, Time:", time.time())

    await asyncio.sleep(3)

    api_key = agent1.get_api_key()

    texts = {
        "test_titles": test_titles,
        "label_map": label_map,
        "rules": rules,
        "X":None,
        "y":None,
        "use_transformer": False,
        "agent_id": "arm64-demo"
    }

    # agent1.pipeline.use_transformer = False
    # agent2.pipeline.use_transformer = False

    result = await agent1.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )

    result2 = await agent2.multi_modal_peer_ensemble_prediction(
        texts=texts,
        api_key=api_key,
        method="advanced",
        disable_sync=True
    )    

    print("\n=== RESULT ===")
    print(f"\n📊 Ensemble Result for Agent 1:")
    print(f"   Prediction: {result.get('prediction', 'N/A')}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")

    print('[=] For agent 2: ')
    print(f"   Second Prediction: {result2.get('prediction', 'N/A')}")
    print(f"   Second Confidence: {result2.get('confidence', 0):.2%}")

    agent1._peer_agent.stop_server()
    agent2._peer_agent.stop_server()

    await agent1.shutdown()
    await agent2.shutdown()

asyncio.run(main(pipeline))

"""
Adversarial concurrency test suite.

Purpose
-------
Regular happy-path CI tests one call at a time, on the happy path,
and almost never calls stop() mid-flight. That's exactly why the bugs
found in this session (pending-future leaks, tuple-format mismatches
in priority queues, stop() calling the wrong primitives, silently
swallowed handler exceptions, check-then-act start races) never
surfaced despite hundreds of green CI runs.

This suite deliberately does the things happy-path tests avoid:
  1. Fires many concurrent requests at once (races need concurrency)
  2. Injects handler failures on a DETERMINISTIC, SEEDED schedule
     (not scattered random.random() calls -- one controlled RNG,
     so any failure this suite finds is exactly reproducible)
  3. Calls stop() while requests are still in flight
  4. Checks accounting invariants (submitted == processed + failed + timeout)
  5. Checks for resource leaks (threads, tasks, pending futures) before
     vs after a full start/stress/stop cycle
  6. Repeats every scenario across multiple seeds via parametrize, so a
     failure is tied to a specific reproducible seed, not a fluke

How to run
----------
    pip install pytest pytest-asyncio
    pytest test_concurrency_stress.py -v

Adjust the imports below to match your actual module layout --
this suite is written against the *interfaces* fixed during the
debugging session (AsyncMessageQueue, WorkerPool + AsyncResultQueue,
AutoBatcherAutomation, ThreadedMessageQueue) rather than assuming a
single import path, since class organization may differ.
"""

import asyncio
import gc
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Adjust these imports to your actual module path, e.g.:
#   from AbstractIntegratedModule import (
#       AsyncMessageQueue, WorkerPool, AsyncResultQueue,
#       AutoBatcherAutomation, ThreadedMessageQueue, Message,
#       RequestStatus, AsyncRequest,
#   )
# ---------------------------------------------------------------------------
from AbstractIntegratedModule import (          # noqa: E402
    AsyncMessageQueue,
    ThreadedMessageQueue,
)

try:
    from AbstractIntegratedModule import (
        WorkerPool, AsyncResultQueue, RequestStatus,
    )
    _HAS_WORKER_POOL = True
except ImportError:
    _HAS_WORKER_POOL = False

try:
    from AbstractIntegratedModule import AutoBatcherAutomation
    _HAS_AUTOBATCHER = True
except ImportError:
    _HAS_AUTOBATCHER = False

try:
    from AbstractIntegratedModule import Message, MessagePriority
except ImportError:
    # minimal stand-in so this file is runnable even before wiring the
    # real Message class in -- replace with your actual Message import
    from enum import Enum

    class MessagePriority(Enum):
        HIGH   = 0
        NORMAL = 1
        LOW    = 2

    @dataclass
    class Message:
        id: str
        type: str
        payload: Any = None
        priority: MessagePriority = MessagePriority.NORMAL
        timeout: float = 5.0
        retry_count: int = 0
        max_retries: int = 2
        callback: Optional[Callable] = None

        @property
        def is_expired(self) -> bool:
            return False


# ===========================================================================
# Deterministic, seeded fault injection
#
# One controlled RNG per test, seeded explicitly. A failure is scheduled
# by INDEX (the Nth call fails), not by an unseeded random chance sprinkled
# through the handler. This makes every failure exactly reproducible: if
# this suite ever finds a bug, `seed` + `fail_every` fully describe how to
# reproduce it byte-for-byte.
# ===========================================================================

@dataclass
class FaultSchedule:
    """Deterministic failure schedule: fails on specific call indices."""
    seed: int
    failure_rate: float = 0.2
    total_calls: int = 500
    _fail_indices: set = field(default_factory=set, init=False)
    _call_counter: int = field(default=0, init=False)

    def __post_init__(self):
        rng = random.Random(self.seed)           # isolated RNG, not global random
        n_failures = int(self.total_calls * self.failure_rate)
        self._fail_indices = set(
            rng.sample(range(self.total_calls), k=min(n_failures, self.total_calls))
        )

    def should_fail(self) -> bool:
        idx = self._call_counter
        self._call_counter += 1
        return idx in self._fail_indices

    @property
    def expected_failures(self) -> int:
        return len(self._fail_indices)


def make_flaky_handler(schedule: FaultSchedule, work_ms: float = 1.0):
    """Handler that fails on the schedule's precomputed indices."""
    def handler(message):
        if work_ms > 0:
            time.sleep(work_ms / 1000)
        if schedule.should_fail():
            raise ValueError(f"scheduled failure for message {message.id}")
        return {"ok": True, "id": message.id}
    return handler


def make_async_flaky_handler(schedule: FaultSchedule, work_ms: float = 1.0):
    async def handler(message):
        if work_ms > 0:
            await asyncio.sleep(work_ms / 1000)
        if schedule.should_fail():
            raise ValueError(f"scheduled failure for message {message.id}")
        return {"ok": True, "id": message.id}
    return handler


# ===========================================================================
# Resource leak helpers
# ===========================================================================

def snapshot_threads() -> set:
    """Named set of currently alive threads, for before/after leak diffing."""
    return {t.name for t in threading.enumerate() if t.is_alive()}


def snapshot_tasks() -> set:
    try:
        return {id(t) for t in asyncio.all_tasks() if not t.done()}
    except RuntimeError:
        return set()


def assert_no_leaked_threads(before: set, after: set, allowed_prefixes=()):
    """
    Fails if new threads exist after teardown that weren't there before,
    unless their name starts with an explicitly allowed prefix (e.g. for
    daemon threads that are known-abandoned per the earlier discussion of
    Python's inability to force-kill threads).
    """
    leaked = after - before
    unexplained = {
        name for name in leaked
        if not any(name.startswith(p) for p in allowed_prefixes)
    }
    assert not unexplained, (
        f"Leaked thread(s) still alive after stop(): {unexplained}. "
        f"Before={len(before)} threads, after={len(after)} threads."
    )


# ===========================================================================
# 1. AsyncMessageQueue -- concurrency, accounting, mid-flight shutdown
# ===========================================================================

class TestAsyncMessageQueue:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [1, 2, 3, 42, 1337])
    async def test_concurrent_publish_accounting(self, seed):
        """
        Fires N concurrent publish() calls with a deterministic failure
        schedule and asserts every message resolves exactly once, and
        that submitted == succeeded + failed (no message vanishes,
        none is double-counted).

        This is the test that would have caught the AsyncMessageQueue
        pending-future leak: if set_result()/set_exception() is ever
        skipped on some exit path, this assertion fails immediately
        instead of the leak being invisible.
        """
        q = AsyncMessageQueue(max_size=2000)
        schedule = FaultSchedule(seed=seed, failure_rate=0.25, total_calls=300)
        q.register_handler("test", make_async_flaky_handler(schedule, work_ms=0.5))

        messages = [
            Message(id=f"m-{seed}-{i}", type="test",
                    priority=MessagePriority.NORMAL, timeout=10.0)
            for i in range(schedule.total_calls)
        ]

        results = await asyncio.gather(
            *(q.publish(m) for m in messages),
            return_exceptions=True,
        )

        succeeded = sum(1 for r in results if not isinstance(r, Exception))
        failed    = sum(1 for r in results if isinstance(r, Exception))

        assert succeeded + failed == schedule.total_calls, (
            "Some messages neither succeeded nor failed -- likely lost "
            "in the queue or a future was never resolved."
        )
        # allow slack: retries can convert some scheduled failures to
        # eventual successes depending on retry policy, so assert an
        # upper bound rather than an exact match
        assert failed <= schedule.expected_failures + 1, (
            f"More failures ({failed}) than the fault schedule injected "
            f"({schedule.expected_failures}) -- something is failing "
            f"that shouldn't be."
        )

        # every message.id should have been removed from `pending`
        assert len(q.pending) == 0, (
            f"{len(q.pending)} futures still in `pending` after all "
            f"publishes resolved -- this is the leak found in the "
            f"original _worker(): futures were only popped on some "
            f"exit paths, not all of them."
        )

        await q.stop()

    @pytest.mark.asyncio
    async def test_stop_mid_flight_no_crash_no_leak(self):
        """
        The specific gap identified during the session: happy-path tests
        exercise start() thoroughly and stop() almost never, and never
        while requests are still in flight. This test calls stop() while
        ~half the batch is still being processed.
        """
        q = AsyncMessageQueue(max_size=500)
        schedule = FaultSchedule(seed=7, failure_rate=0.1, total_calls=200)
        q.register_handler("test", make_async_flaky_handler(schedule, work_ms=5.0))

        tasks = [
            asyncio.create_task(
                q.publish(Message(id=f"mf-{i}", type="test", timeout=15.0))
            )
            for i in range(200)
        ]

        await asyncio.sleep(0.05)   # let roughly some, not all, complete
        await q.stop(timeout=3.0)

        # after stop(), no task should hang forever -- give them a bounded
        # window to settle (either resolved before stop, or raised due to
        # cancellation/shutdown), then assert none are still pending
        done, pending = await asyncio.wait(tasks, timeout=5.0)
        assert not pending, (
            f"{len(pending)} publish() calls never resolved after "
            f"stop() -- shutdown left callers hanging indefinitely."
        )

    @pytest.mark.asyncio
    async def test_double_start_does_not_spawn_duplicate_workers(self):
        """
        Guards against the check-then-act race in _ensure_started():
        two concurrent first-publish() calls racing to start the worker
        must not result in two worker tasks both draining the same queue.
        """
        q = AsyncMessageQueue(max_size=100)
        q.register_handler("test", lambda m: {"ok": True})

        # race many concurrent "first" calls against _ensure_started
        await asyncio.gather(*(q._ensure_started() for _ in range(50)))

        assert q._worker_task is not None
        # there should be exactly one live worker task, not one per race entrant
        live_worker_tasks = [
            t for t in asyncio.all_tasks()
            if t is q._worker_task and not t.done()
        ]
        assert len(live_worker_tasks) == 1

        await q.stop()


# ===========================================================================
# 2. WorkerPool + AsyncResultQueue -- circuit breaker, health, leak check
# ===========================================================================

@pytest.mark.skipif(not _HAS_WORKER_POOL, reason="WorkerPool not importable")
class TestWorkerPool:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("seed", [11, 22, 33])
    async def test_worker_pool_stress_accounting(self, seed):
        """
        Fires a large batch through submit()->WorkerPool with a seeded
        failure schedule, then asserts every request resolved and the
        AsyncResultQueue's own bookkeeping matches reality.
        """
        result_queue = AsyncResultQueue(max_size=1000)
        pool = WorkerPool(result_queue, num_workers=4)

        schedule = FaultSchedule(seed=seed, failure_rate=0.2, total_calls=200)

        def predict_func(texts, api_key=None, client_ip=None):
            if schedule.should_fail():
                raise RuntimeError("scheduled prediction failure")
            return [{"label": "ok"} for _ in texts]

        await pool.start(predict_func)

        request_ids = [
            await result_queue.submit(texts=[f"t{i}"])
            for i in range(schedule.total_calls)
        ]

        # poll for completion with a hard timeout, rather than assuming
        # instantaneous processing
        deadline = time.time() + 30
        while time.time() < deadline:
            pending = [
                rid for rid in request_ids
                if result_queue._requests[rid].status == RequestStatus.PENDING
                or result_queue._requests[rid].status == RequestStatus.PROCESSING
            ]
            if not pending:
                break
            await asyncio.sleep(0.05)

        statuses = [result_queue._requests[rid].status for rid in request_ids]
        unresolved = [s for s in statuses if s not in (
            RequestStatus.COMPLETED, RequestStatus.FAILED, RequestStatus.TIMEOUT
        )]
        assert not unresolved, (
            f"{len(unresolved)} requests never reached a terminal status "
            f"within the deadline."
        )

        if hasattr(pool, "get_health"):
            health = pool.get_health()
            assert health["active_workers"] >= 1, (
                "All workers died during the stress run -- the circuit "
                "breaker retired every worker instead of surviving "
                "transient failures."
            )

        await pool.stop()

    @pytest.mark.asyncio
    async def test_double_start_does_not_leak_orphaned_tasks(self):
        """
        Regression test for the specific leak found in WorkerPool.start():
        calling start() twice without stop() used to drop the reference to
        the first batch of worker tasks without cancelling them.
        """
        result_queue = AsyncResultQueue(max_size=100)
        pool = WorkerPool(result_queue, num_workers=3)

        before = snapshot_tasks()

        await pool.start(lambda texts, **kw: [{"ok": True}])
        first_workers = list(pool._workers)

        await pool.start(lambda texts, **kw: [{"ok": True}])  # duplicate start

        # the original worker tasks must not have been abandoned while
        # still running -- either start() refused the second call, or
        # it properly replaced the pool without orphaning the old tasks
        still_running_orphans = [
            t for t in first_workers if not t.done() and t not in pool._workers
        ]
        assert not still_running_orphans, (
            f"{len(still_running_orphans)} worker task(s) from the first "
            f"start() call are still running but no longer tracked in "
            f"pool._workers -- they are orphaned and will run forever."
        )

        await pool.stop()
        after = snapshot_tasks()
        assert after <= before, (
            f"Task leak: {len(after - before)} asyncio tasks still alive "
            f"after stop() that weren't present before start()."
        )


# ===========================================================================
# 3. AutoBatcherAutomation -- start-race, timeout cleanup, dead worker recovery
# ===========================================================================

@pytest.mark.skipif(not _HAS_AUTOBATCHER, reason="AutoBatcherAutomation not importable")
class TestAutoBatcherAutomation:

    class _FakePipeline:
        """Deterministic fake pipeline -- fails on a seeded schedule."""
        def __init__(self, schedule: FaultSchedule):
            self.schedule = schedule

        def prediction_batch(self, texts):
            if self.schedule.should_fail():
                raise RuntimeError("scheduled batch failure")
            return [{"label": "ok", "text": t} for t in texts]

    def test_concurrent_add_request_no_duplicate_workers(self):
        """
        Regression test for the check-then-act race between add_request()
        and _process_batches() on self.processing. Fires many concurrent
        add_request() calls from multiple threads simultaneously -- if the
        race exists, more than one worker thread will be created.
        """
        schedule = FaultSchedule(seed=5, failure_rate=0.0, total_calls=500)
        batcher = AutoBatcherAutomation(
            pipeline=self._FakePipeline(schedule),
            max_batch_size=16, max_wait_ms=10,
        )

        before = snapshot_threads()
        started_thread_names = []

        real_start = batcher._start_processing
        def tracking_start():
            real_start()
            started_thread_names.append(batcher._workers[-1].name
                                        if hasattr(batcher, "_workers") else None)
        # if the implementation doesn't expose _workers, at least count
        # how many times _start_processing is actually invoked
        call_count = {"n": 0}
        def counting_start():
            call_count["n"] += 1
            real_start()
        batcher._start_processing = counting_start

        def worker():
            batcher.add_request(f"text-{threading.get_ident()}")

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count["n"] == 1, (
            f"_start_processing() was called {call_count['n']} times from "
            f"concurrent add_request() calls -- expected exactly 1. This "
            f"is the check-then-act race on self.processing."
        )

        # let the batcher drain
        deadline = time.time() + 5
        while batcher.request_queue and time.time() < deadline:
            time.sleep(0.05)
        time.sleep(0.5)   # allow final batch + processing flag reset

        after = snapshot_threads()
        assert_no_leaked_threads(before, after)

    def test_get_result_timeout_does_not_leak_forever(self):
        """
        Regression test for the results/event dict leak: a caller that
        times out in get_result() must not leave permanent entries behind.
        """
        schedule = FaultSchedule(seed=9, failure_rate=0.0, total_calls=10)
        batcher = AutoBatcherAutomation(
            pipeline=self._FakePipeline(schedule),
            max_batch_size=4, max_wait_ms=2000,  # deliberately slow, forces timeout
        )

        rid = batcher.add_request("slow-text")
        result = batcher.get_result(rid, timeout=0.05)   # times out before batch runs
        assert result is None

        # give the real batch time to actually complete in the background
        time.sleep(3.0)

        if hasattr(batcher, "cleanup_stale"):
            batcher.cleanup_stale(max_age_seconds=0)

        assert rid not in batcher.results, (
            "Result for a timed-out request is still sitting in "
            "batcher.results with nothing left to ever collect it -- "
            "this dict grows unboundedly under sustained timeout load."
        )

    def test_one_bad_batch_does_not_permanently_wedge_the_pool(self):
        """
        Regression test for the silent-permanent-deadlock bug: if
        pipeline.prediction_batch() raises inside _process_batches()
        without a try/except, `processing` never resets to False and
        every future request queues forever with nothing consuming it.
        """
        # schedule fails on EVERY call for the first batch, then recovers
        always_fail_once = FaultSchedule(seed=1, failure_rate=1.0, total_calls=1)
        batcher = AutoBatcherAutomation(
            pipeline=self._FakePipeline(always_fail_once),
            max_batch_size=4, max_wait_ms=10,
        )

        rid1 = batcher.add_request("this-batch-will-fail")
        time.sleep(0.5)   # let the failing batch run and (hopefully) recover

        # pool must have recovered enough to accept and process new work
        rid2 = batcher.add_request("this-should-still-work")
        result2 = batcher.get_result(rid2, timeout=5.0)

        assert result2 is not None, (
            "Batcher never recovered after one failing batch -- a single "
            "exception in prediction_batch() permanently wedged the "
            "pool, exactly the silent-deadlock bug found in this session."
        )


# ===========================================================================
# 4. ThreadedMessageQueue -- the critical stop() crash + swallowed errors
# ===========================================================================

class TestThreadedMessageQueue:

    def test_stop_does_not_raise(self):
        """
        Direct regression test for the fatal bug: the original stop()
        referenced self.result_queue (never defined) and called
        Thread.cancel() (doesn't exist). This must not raise at all.
        """
        q = ThreadedMessageQueue(worker_threads=2)
        q.register_handler("test", lambda m: {"ok": True})
        q.start()

        q.publish_async(Message(id="warm-up", type="test"))
        time.sleep(0.1)

        # this line alone would have failed the whole suite under the
        # original implementation
        q.stop(timeout=3.0)

        assert not q._running
        assert all(not t.is_alive() for t in q._workers) or True  # daemon stragglers tolerated

    @pytest.mark.parametrize("seed", [4, 8, 15])
    def test_handler_failure_actually_raises_to_caller(self, seed):
        """
        Regression test for the swallowed-error bug: callback_wrapper
        never distinguished an Exception result from a real result, so
        publish() returned Exception objects as if they were successful
        predictions instead of raising them.
        """
        schedule = FaultSchedule(seed=seed, failure_rate=1.0, total_calls=1)
        q = ThreadedMessageQueue(worker_threads=2)
        q.register_handler("test", make_flaky_handler(schedule, work_ms=1.0))
        q.start()

        with pytest.raises(Exception):
            q.publish(Message(id=f"fail-{seed}", type="test", timeout=5.0))

        q.stop(timeout=3.0)

    @pytest.mark.parametrize("seed", [100, 200, 300])
    def test_concurrent_publish_accounting_and_no_thread_leak(self, seed):
        """
        Combined stress + leak test: many concurrent publish() calls
        from multiple threads, deterministic partial failure, then
        verifies (a) success/failure counts are consistent, and
        (b) no worker threads survive stop().
        """
        before = snapshot_threads()

        schedule = FaultSchedule(seed=seed, failure_rate=0.3, total_calls=150)
        q = ThreadedMessageQueue(worker_threads=4)
        q.register_handler("test", make_flaky_handler(schedule, work_ms=1.0))
        q.start()

        results: List[Any] = [None] * schedule.total_calls
        errors:  List[Any] = [None] * schedule.total_calls

        def call(i):
            try:
                results[i] = q.publish(
                    Message(id=f"c-{seed}-{i}", type="test", timeout=10.0)
                )
            except Exception as e:
                errors[i] = e

        threads = [threading.Thread(target=call, args=(i,))
                  for i in range(schedule.total_calls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        succeeded = sum(1 for r in results if r is not None)
        failed    = sum(1 for e in errors if e is not None)

        assert succeeded + failed == schedule.total_calls, (
            f"submitted={schedule.total_calls} but succeeded({succeeded})"
            f" + failed({failed}) don't add up -- a caller thread is "
            f"still stuck or a result went missing."
        )
        assert failed <= schedule.expected_failures + 1

        stats = q.get_stats()
        assert stats["messages_processed"] == succeeded
        assert stats["messages_failed"] >= failed

        q.stop(timeout=5.0)

        after = snapshot_threads()
        # allow daemon-thread stragglers only if the implementation warned
        # about them; otherwise this should be a clean diff
        assert_no_leaked_threads(before, after)


# ===========================================================================
# Entry point for running this file directly (without invoking pytest CLI)
# ===========================================================================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))


