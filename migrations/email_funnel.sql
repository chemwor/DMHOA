-- Email funnel state. One row per email address. Stages advance forward only.
create table email_funnel (
  id uuid primary key default gen_random_uuid(),
  email text unique not null,
  stage text not null,
  stage_completed_at timestamptz not null default now(),
  nudge_1_sent boolean not null default false,
  nudge_2_sent boolean not null default false,
  nudge_3_sent boolean not null default false,
  purchased boolean not null default false,
  created_at timestamptz not null default now()
);

create index on email_funnel (stage);
create index on email_funnel (purchased);
create index on email_funnel (stage_completed_at);
