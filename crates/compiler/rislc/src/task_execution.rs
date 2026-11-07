//! Jobserver-aware parallel task execution.

use std::any::Any;
use std::env;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, ScopedJoinHandle};

use crossbeam_channel::{Receiver, Sender, TrySendError, bounded};
use jobserver::{Acquired, Client, FromEnvErrorKind};

enum Budget {
    Inherited(Client),
    Standalone,
}

pub struct ExecutionContext {
    workers: usize,
    budget: Budget,
}

impl ExecutionContext {
    /// Creates a new execution context.
    ///
    /// # Safety
    ///
    /// Connect before rustc starts threads or opens files, see [jobserver::Client::from_env_ext].
    pub unsafe fn from_env() -> Self {
        let threads = env::var_os("RISL_CODEGEN_THREADS")
            .map(|value| {
                value
                    .to_str()
                    .and_then(|value| value.parse::<usize>().ok())
                    .filter(|&value| value > 0)
                    .expect("RISL_CODEGEN_THREADS must be a positive integer")
            })
            .unwrap_or_else(|| thread::available_parallelism().map_or(1, usize::from));

        // We reserve one thread for the main rislc process, the rest are available for workers.
        let workers = threads - 1;

        // SAFETY: the caller guarantees that inherited descriptors have not been repurposed.
        let inherited = unsafe { Client::from_env_ext(true) };

        match inherited.client {
            Ok(client) => ExecutionContext {
                workers,
                budget: Budget::Inherited(client),
            },
            Err(error)
                if matches!(
                    error.kind(),
                    FromEnvErrorKind::NoEnvVar | FromEnvErrorKind::NoJobserver
                ) =>
            {
                ExecutionContext {
                    workers,
                    budget: Budget::Standalone,
                }
            }
            Err(_) => ExecutionContext {
                workers: 0,
                budget: Budget::Standalone,
            },
        }
    }

    pub fn run<T, R>(
        &self,
        task_count: usize,
        execute: impl Fn(T) + Sync,
        schedule: impl FnOnce(&mut Executor<'_, T>) -> R,
    ) -> R
    where
        T: Send,
    {
        let workers = self.workers.min(task_count);
        let failure = Failure::default();

        thread::scope(|scope| {
            let (sender, receiver) = bounded(workers);
            let stack_size = worker_stack_size();
            let execute = &execute;
            let failure = &failure;

            let mut handles = Vec::with_capacity(workers);

            for index in 0..workers {
                let receiver = receiver.clone();
                let handle = thread::Builder::new()
                    .name(format!("risl-codegen-{index}"))
                    .stack_size(stack_size)
                    .spawn_scoped(scope, move || worker(receiver, execute, failure))
                    .expect("failed to spawn shader codegen worker");

                handles.push(handle);
            }

            drop(receiver);

            let mut pipeline = Executor {
                sender: Some(sender),
                handles,
                failure,
                budget: &self.budget,
                execute,
            };
            let result = schedule(&mut pipeline);

            pipeline.finish();

            result
        })
    }
}

fn worker_stack_size() -> usize {
    env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
        .max(8 * 1024 * 1024)
}

type Panic = Box<dyn Any + Send + 'static>;
type Work<T> = (T, Option<Acquired>);

#[derive(Default)]
struct Failure {
    cancelled: AtomicBool,
    panic: Mutex<Option<Panic>>,
}

impl Failure {
    fn propagate(&self) {
        let panic = self.panic.lock().unwrap().take();

        if let Some(panic) = panic {
            resume_unwind(panic);
        }
    }
}

fn worker<T>(receiver: Receiver<Work<T>>, execute: &impl Fn(T), failure: &Failure) {
    while let Ok(work) = receiver.recv() {
        if failure.cancelled.load(Ordering::Acquire) {
            break;
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let (task, token) = work;

            execute(task);
            drop(token);
        }));

        if let Err(panic) = result {
            let mut first_panic = failure.panic.lock().unwrap();

            if first_panic.is_none() {
                *first_panic = Some(panic);
            }

            failure.cancelled.store(true, Ordering::Release);

            break;
        }
    }
}

pub struct Executor<'a, T> {
    sender: Option<Sender<Work<T>>>,
    handles: Vec<ScopedJoinHandle<'a, ()>>,
    failure: &'a Failure,
    budget: &'a Budget,
    execute: &'a dyn Fn(T),
}

impl<T> Executor<'_, T> {
    pub fn submit(&mut self, task: T) {
        self.failure.propagate();

        if !self.handles.is_empty() {
            let token = match self.budget {
                Budget::Inherited(client) => match client.try_acquire() {
                    Ok(Some(token)) => Some(token),
                    // If we can't acquire a token, whether because all slots are already occupied
                    // or because the jobserver is not available on the current platform, then we
                    // execute the task on the main thread.
                    _ => return (self.execute)(task),
                },
                Budget::Standalone => None,
            };

            match self.sender.as_ref().unwrap().try_send((task, token)) {
                Ok(()) => return,
                Err(
                    TrySendError::Full((task, token)) | TrySendError::Disconnected((task, token)),
                ) => {
                    drop(token);
                    self.failure.propagate();
                    (self.execute)(task);
                }
            }
        } else {
            (self.execute)(task);
        }
    }

    fn join(&mut self) {
        self.sender.take();

        for handle in self.handles.drain(..) {
            if let Err(panic) = handle.join() {
                let mut first_panic = self.failure.panic.lock().unwrap();

                if first_panic.is_none() {
                    *first_panic = Some(panic);
                }
            }
        }
    }

    fn finish(mut self) {
        self.join();
        self.failure.propagate();
    }
}

impl<T> Drop for Executor<'_, T> {
    fn drop(&mut self) {
        self.failure.cancelled.store(true, Ordering::Release);
        self.join();
    }
}
