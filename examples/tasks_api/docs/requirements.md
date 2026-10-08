# Tasks API requirements

## Ownership and visibility
- Every task has exactly one owner, the user who created it.
- A user can only see, edit, complete or delete their own tasks. Asking for another user's task
  must look exactly like asking for a task that does not exist: respond 404, never 403 and never
  the task itself.
- An admin can see every task, and the task list shows all of them.

## Listing
- Tasks are listed in creation order, 20 per page by default, at most 100 per page.
- Page 2 starts with the task that follows the last task of page 1. No task is skipped or repeated.
- The `status` filter accepts only `open` and `done`. Any other value is rejected with 422.

## Creating and editing
- A title is required and must be 1 to 100 characters.
- Priority is `low`, `medium` or `high` and defaults to `medium`.
- A due date in the past is rejected with 422.
- A completed task can no longer be edited: respond 409.

## Completing
- Completing an open task sets its status to `done`.
- Completing a task that is already done is a conflict: respond 409. It must not succeed twice.
