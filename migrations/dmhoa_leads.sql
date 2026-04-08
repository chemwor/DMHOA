create table dmhoa_leads (
  id uuid primary key default gen_random_uuid(),
  post_id text unique not null,
  subreddit text,
  title text,
  url text,
  score int default 0,
  status text default 'new',
  replied_at timestamptz,
  created_utc timestamptz,
  inserted_at timestamptz default now()
);

create index on dmhoa_leads (status);
create index on dmhoa_leads (score desc);
