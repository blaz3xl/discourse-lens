from __future__ import annotations

import sqlite3

from .errors import MigrationError
from .revision_provenance import basic_extraction_manifest

SCHEMA_VERSION = 52
DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS project_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, display_name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS document_revisions(id TEXT PRIMARY KEY,document_id TEXT NOT NULL,source_hash TEXT NOT NULL,canonical_hash TEXT NOT NULL,extractor TEXT NOT NULL,coordinate_scheme TEXT NOT NULL,text_path TEXT NOT NULL,original_path TEXT,mime_type TEXT,source_size INTEGER,extraction_status TEXT NOT NULL DEFAULT 'SUCCESS',created_at TEXT NOT NULL,ocr_language TEXT,ocr_dpi INTEGER,extraction_manifest_json TEXT NOT NULL DEFAULT '{}',structure_hash TEXT NOT NULL DEFAULT '',UNIQUE(document_id,source_hash,canonical_hash,extractor,coordinate_scheme,extraction_manifest_json,structure_hash));
CREATE TABLE IF NOT EXISTS paragraphs(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,ordinal INTEGER NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,page_number INTEGER,page_id TEXT,text TEXT,segmentation_version TEXT,bounding_boxes_json TEXT,UNIQUE(revision_id,ordinal));
CREATE TABLE IF NOT EXISTS sentences(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,paragraph_id TEXT NOT NULL,ordinal INTEGER NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,text TEXT NOT NULL,segmentation_version TEXT NOT NULL,UNIQUE(revision_id,ordinal));
CREATE TABLE IF NOT EXISTS source_spans(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,text_hash TEXT NOT NULL,UNIQUE(revision_id,start_offset,end_offset));
CREATE TABLE IF NOT EXISTS source_span_segments(id TEXT PRIMARY KEY,span_id TEXT NOT NULL,paragraph_id TEXT,page_number INTEGER,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL,bounding_boxes_json TEXT,bbox_status TEXT NOT NULL DEFAULT 'UNAVAILABLE',bbox_projection_version TEXT NOT NULL DEFAULT 'none-v1');
CREATE TABLE IF NOT EXISTS pages(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,page_index INTEGER NOT NULL,page_label TEXT,width REAL,height REAL,rotation INTEGER NOT NULL,canonical_start INTEGER NOT NULL,canonical_end INTEGER NOT NULL,text TEXT NOT NULL,UNIQUE(revision_id,page_index));
CREATE TABLE IF NOT EXISTS extraction_warnings(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,code TEXT NOT NULL,detail TEXT);
CREATE TABLE IF NOT EXISTS evidence_artifacts(id TEXT PRIMARY KEY,kind TEXT NOT NULL,span_id TEXT,content_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS findings(id TEXT PRIMARY KEY,analysis_run_id TEXT NOT NULL,skill_run_id TEXT,skill_id TEXT NOT NULL,skill_version TEXT NOT NULL,phenomenon TEXT NOT NULL,observation TEXT NOT NULL,interpretation TEXT,alternative_interpretation TEXT,claim_types_json TEXT NOT NULL,coverage_id TEXT,evidence_sufficiency TEXT NOT NULL,interpretation_confidence TEXT NOT NULL,limitations_json TEXT NOT NULL,review_status TEXT NOT NULL,created_at TEXT NOT NULL,finalized_at TEXT);
CREATE TABLE IF NOT EXISTS finding_observation_candidates(id TEXT PRIMARY KEY,finding_id TEXT NOT NULL,ordinal INTEGER NOT NULL,evidence_id TEXT NOT NULL,excerpt TEXT NOT NULL,description TEXT NOT NULL,status TEXT NOT NULL CHECK(status='UNVERIFIED'),created_at TEXT NOT NULL,UNIQUE(finding_id,ordinal));
CREATE TABLE IF NOT EXISTS finding_lineage(finding_id TEXT NOT NULL,source_finding_id TEXT NOT NULL,relation TEXT NOT NULL,PRIMARY KEY(finding_id,source_finding_id,relation));
CREATE TABLE IF NOT EXISTS finding_reviews(id TEXT PRIMARY KEY,finding_id TEXT NOT NULL,status TEXT NOT NULL,note TEXT,created_at TEXT NOT NULL,review_order INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS finding_evidence(finding_id TEXT NOT NULL,evidence_id TEXT NOT NULL,role TEXT NOT NULL,rationale TEXT,PRIMARY KEY(finding_id,evidence_id,role));
CREATE TABLE IF NOT EXISTS scope_snapshots(id TEXT PRIMARY KEY,revision_ids_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS coverage_descriptors(id TEXT PRIMARY KEY,payload_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS retrieval_runs(id TEXT PRIMARY KEY,analysis_run_id TEXT,skill_run_id TEXT,scope_id TEXT NOT NULL,retrieval_plan_ref TEXT NOT NULL,query_terms_json TEXT NOT NULL,selection_policy TEXT NOT NULL,candidate_count INTEGER NOT NULL,inspected_count INTEGER NOT NULL,returned_count INTEGER NOT NULL,candidate_limit_reached INTEGER NOT NULL,coverage_id TEXT NOT NULL,created_at TEXT NOT NULL,CHECK((analysis_run_id IS NULL)=(skill_run_id IS NULL)));
CREATE TABLE IF NOT EXISTS retrieval_run_context(retrieval_run_id TEXT PRIMARY KEY,operation TEXT NOT NULL,config_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS retrieval_hits(id TEXT PRIMARY KEY,retrieval_run_id TEXT NOT NULL,source_span_id TEXT NOT NULL,retrieval_method TEXT NOT NULL,rank INTEGER NOT NULL,score REAL NOT NULL,UNIQUE(retrieval_run_id,source_span_id));
CREATE TABLE IF NOT EXISTS retrieval_hit_context(retrieval_hit_id TEXT PRIMARY KEY,artifact_id TEXT,unit_id TEXT NOT NULL,lexical_rank INTEGER,vector_rank INTEGER,CHECK(lexical_rank IS NULL OR lexical_rank>0),CHECK(vector_rank IS NULL OR vector_rank>0),CHECK((artifact_id IS NULL)=(vector_rank IS NULL)));
CREATE TABLE IF NOT EXISTS reranking_runs(id TEXT PRIMARY KEY,retrieval_run_id TEXT NOT NULL,provider_id TEXT NOT NULL,credential_alias TEXT,model TEXT NOT NULL,query TEXT NOT NULL,top_n INTEGER,effective_top_n INTEGER,candidate_count INTEGER NOT NULL,returned_count INTEGER NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reranking_hits(id TEXT PRIMARY KEY,reranking_run_id TEXT NOT NULL,retrieval_hit_id TEXT NOT NULL,rank INTEGER NOT NULL,relevance_score REAL NOT NULL,UNIQUE(reranking_run_id,retrieval_hit_id),UNIQUE(reranking_run_id,rank));
CREATE TABLE IF NOT EXISTS comparison_runs(id TEXT PRIMARY KEY,left_scope_id TEXT NOT NULL,right_scope_id TEXT NOT NULL,metric TEXT NOT NULL,calculation_version TEXT NOT NULL,config_json TEXT NOT NULL,result_json TEXT NOT NULL,result_hash TEXT NOT NULL,created_at TEXT NOT NULL,storage_kind TEXT NOT NULL DEFAULT 'json-v1');
CREATE TABLE IF NOT EXISTS comparison_result_rows(comparison_id TEXT NOT NULL,token TEXT NOT NULL,rank_value REAL NOT NULL,row_json TEXT NOT NULL,PRIMARY KEY(comparison_id,token));
CREATE INDEX IF NOT EXISTS comparison_result_rows_rank ON comparison_result_rows(comparison_id,rank_value DESC,token COLLATE BINARY ASC);
CREATE TABLE IF NOT EXISTS codebooks(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,description TEXT,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS codes(id TEXT PRIMARY KEY,codebook_id TEXT NOT NULL,parent_code_id TEXT,label TEXT NOT NULL,definition TEXT,created_at TEXT NOT NULL,UNIQUE(codebook_id,label));
CREATE TABLE IF NOT EXISTS annotations(id TEXT PRIMARY KEY,code_id TEXT NOT NULL,span_id TEXT NOT NULL,memo TEXT,created_at TEXT NOT NULL,UNIQUE(code_id,span_id));
CREATE TABLE IF NOT EXISTS research_memos(id TEXT PRIMARY KEY,scope_id TEXT NOT NULL,title TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_runs(id TEXT PRIMARY KEY,scope_id TEXT NOT NULL,status TEXT NOT NULL,config_json TEXT NOT NULL,config_hash TEXT NOT NULL,started_at TEXT NOT NULL,completed_at TEXT,software_version TEXT NOT NULL DEFAULT 'unknown',project_format_version INTEGER NOT NULL DEFAULT 0,database_schema_version INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS skill_runs(id TEXT PRIMARY KEY,analysis_run_id TEXT NOT NULL,skill_id TEXT NOT NULL,skill_version TEXT NOT NULL,scope_id TEXT NOT NULL,status TEXT NOT NULL,executor TEXT NOT NULL,config_hash TEXT NOT NULL,coverage_id TEXT,input_finding_id TEXT,started_at TEXT,completed_at TEXT,error_type TEXT,skip_reason TEXT);
CREATE TABLE IF NOT EXISTS artifact_registry(id TEXT PRIMARY KEY,revision_id TEXT NOT NULL,artifact_type TEXT NOT NULL,backend TEXT NOT NULL,backend_version TEXT,model TEXT,model_version TEXT,schema_version TEXT NOT NULL,content_hash TEXT NOT NULL,relative_path TEXT NOT NULL,created_at TEXT NOT NULL,file_hash TEXT,embedding_provider_id TEXT,embedding_model TEXT,embedding_dimension INTEGER,chunker_version TEXT,embedding_input_type TEXT);
CREATE TABLE IF NOT EXISTS provider_credentials(id TEXT PRIMARY KEY,provider_id TEXT NOT NULL,credential_alias TEXT NOT NULL,secret_reference TEXT NOT NULL,endpoint TEXT,created_at TEXT NOT NULL,UNIQUE(provider_id,credential_alias));
CREATE TABLE IF NOT EXISTS encrypted_secrets(secret_reference TEXT PRIMARY KEY,ciphertext BLOB NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_presets(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,provider_id TEXT NOT NULL,credential_alias TEXT NOT NULL,model TEXT NOT NULL,reasoning TEXT,reasoning_mode TEXT,reasoning_budget_tokens INTEGER,parameters_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS model_prices(id TEXT PRIMARY KEY,provider_id TEXT NOT NULL,model TEXT NOT NULL,input_per_million_tokens TEXT NOT NULL,output_per_million_tokens TEXT NOT NULL,reasoning_per_million_tokens TEXT,currency TEXT NOT NULL,source_reference TEXT NOT NULL,rate_version TEXT NOT NULL,rate_date TEXT NOT NULL,created_at TEXT NOT NULL,UNIQUE(provider_id,model,rate_version));
CREATE TABLE IF NOT EXISTS model_capability_profiles(id TEXT PRIMARY KEY,provider_id TEXT NOT NULL,model TEXT NOT NULL,capabilities_json TEXT NOT NULL,assurance TEXT NOT NULL,source_reference TEXT,updated_at TEXT NOT NULL,UNIQUE(provider_id,model));
CREATE TABLE IF NOT EXISTS model_routing_profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,routes_json TEXT NOT NULL,fallback_preset TEXT,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_profiles(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,payload_json TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chat_conversations(id TEXT PRIMARY KEY,title TEXT NOT NULL,revision_ids_json TEXT NOT NULL,framework_id TEXT NOT NULL,preset_name TEXT NOT NULL,retrieval_mode TEXT NOT NULL,embedding_preset_name TEXT,embedding_artifact_ids_json TEXT NOT NULL DEFAULT '[]',created_at TEXT NOT NULL,updated_at TEXT NOT NULL,research_project_id TEXT);
CREATE TABLE IF NOT EXISTS research_projects(id TEXT PRIMARY KEY,name TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS research_project_documents(project_id TEXT NOT NULL,document_id TEXT NOT NULL,added_at TEXT NOT NULL,PRIMARY KEY(project_id,document_id),FOREIGN KEY(project_id) REFERENCES research_projects(id) ON DELETE CASCADE,FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS chat_messages(id TEXT PRIMARY KEY,conversation_id TEXT NOT NULL,ordinal INTEGER NOT NULL,role TEXT NOT NULL CHECK(role IN ('user','assistant')),content_json TEXT NOT NULL,analysis_run_id TEXT,finding_id TEXT,created_at TEXT NOT NULL,UNIQUE(conversation_id,ordinal),FOREIGN KEY(conversation_id) REFERENCES chat_conversations(id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS model_executions(id TEXT PRIMARY KEY,analysis_run_id TEXT,provider_id TEXT NOT NULL,credential_alias TEXT NOT NULL,model TEXT NOT NULL,reasoning TEXT,reasoning_mode TEXT,reasoning_budget_tokens INTEGER,request_hash TEXT NOT NULL,response_hash TEXT,error_type TEXT,input_tokens INTEGER,output_tokens INTEGER,reasoning_tokens INTEGER,latency_ms INTEGER,started_at TEXT NOT NULL,completed_at TEXT,parameters_json TEXT NOT NULL DEFAULT '{}',capabilities_json TEXT NOT NULL DEFAULT '{}',model_price_json TEXT,estimated_cost TEXT,estimated_cost_currency TEXT,endpoint TEXT,skill_run_id TEXT);
CREATE TABLE IF NOT EXISTS corpora(id TEXT PRIMARY KEY,name TEXT NOT NULL UNIQUE,description TEXT,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS corpus_memberships(corpus_id TEXT NOT NULL,revision_id TEXT NOT NULL,added_at TEXT NOT NULL,PRIMARY KEY(corpus_id,revision_id));
CREATE TABLE IF NOT EXISTS subcorpora(id TEXT PRIMARY KEY,corpus_id TEXT NOT NULL,name TEXT NOT NULL,description TEXT,created_at TEXT NOT NULL,UNIQUE(corpus_id,name));
CREATE TABLE IF NOT EXISTS subcorpus_memberships(subcorpus_id TEXT NOT NULL,revision_id TEXT NOT NULL,PRIMARY KEY(subcorpus_id,revision_id));
CREATE TABLE IF NOT EXISTS document_metadata(document_id TEXT NOT NULL,key TEXT NOT NULL,value_json TEXT NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(document_id,key));
CREATE VIRTUAL TABLE IF NOT EXISTS revision_fts USING fts5(revision_id UNINDEXED,paragraph_id UNINDEXED,text);
"""

IMMUTABILITY_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS scope_snapshots_immutable_insert_conflict
       BEFORE INSERT ON scope_snapshots
       WHEN EXISTS(SELECT 1 FROM scope_snapshots WHERE id=NEW.id)
       BEGIN SELECT RAISE(ABORT,'ScopeSnapshot records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS scope_snapshots_immutable_update
       BEFORE UPDATE ON scope_snapshots
       BEGIN SELECT RAISE(ABORT,'ScopeSnapshot records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS scope_snapshots_immutable_delete
       BEFORE DELETE ON scope_snapshots
       BEGIN SELECT RAISE(ABORT,'ScopeSnapshot records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS coverage_descriptors_immutable_insert_conflict
       BEFORE INSERT ON coverage_descriptors
       WHEN EXISTS(SELECT 1 FROM coverage_descriptors WHERE id=NEW.id)
       BEGIN SELECT RAISE(ABORT,'CoverageDescriptor records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS coverage_descriptors_immutable_update
       BEFORE UPDATE ON coverage_descriptors
       BEGIN SELECT RAISE(ABORT,'CoverageDescriptor records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS coverage_descriptors_immutable_delete
       BEFORE DELETE ON coverage_descriptors
       BEGIN SELECT RAISE(ABORT,'CoverageDescriptor records are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS analysis_runs_provenance_immutable_insert_conflict
       BEFORE INSERT ON analysis_runs
       WHEN EXISTS(SELECT 1 FROM analysis_runs WHERE id=NEW.id)
       BEGIN SELECT RAISE(ABORT,'analysis run identity and provenance are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS analysis_runs_provenance_immutable
       BEFORE UPDATE OF id,scope_id,config_json,config_hash,started_at,software_version,
           project_format_version,database_schema_version ON analysis_runs
       BEGIN SELECT RAISE(ABORT,'analysis run identity and provenance are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS analysis_runs_lifecycle_transition
       BEFORE UPDATE OF status,completed_at ON analysis_runs
       WHEN NOT (
           (NEW.status IS OLD.status AND NEW.completed_at IS OLD.completed_at)
           OR (OLD.status='PENDING' AND NEW.status='RUNNING'
               AND NEW.completed_at IS OLD.completed_at)
           OR (OLD.status='FAILED' AND NEW.status='RUNNING' AND NEW.completed_at IS NULL)
           OR (OLD.status IN ('PENDING','RUNNING') AND NEW.status IN ('FAILED','COMPLETED')
               AND NEW.completed_at IS NOT NULL)
       )
       BEGIN SELECT RAISE(ABORT,'invalid analysis run lifecycle transition'); END""",
    """CREATE TRIGGER IF NOT EXISTS analysis_runs_immutable_delete
       BEFORE DELETE ON analysis_runs
       BEGIN SELECT RAISE(ABORT,'analysis run history is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS skill_runs_immutable_insert_conflict
       BEFORE INSERT ON skill_runs
       WHEN EXISTS(SELECT 1 FROM skill_runs WHERE id=NEW.id)
       BEGIN SELECT RAISE(ABORT,'SkillRun identity and provenance are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS skill_runs_provenance_immutable
       BEFORE UPDATE OF id,analysis_run_id,skill_id,skill_version,scope_id,executor,config_hash,
           input_finding_id,started_at ON skill_runs
       BEGIN SELECT RAISE(ABORT,'SkillRun identity and provenance are immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS skill_runs_coverage_update_requires_running
       BEFORE UPDATE OF coverage_id ON skill_runs
       WHEN NEW.coverage_id IS NOT OLD.coverage_id
         AND (OLD.status!='RUNNING' OR NEW.status!='RUNNING'
              OR OLD.coverage_id IS NOT NULL OR NEW.coverage_id IS NULL)
       BEGIN SELECT RAISE(ABORT,'SkillRun coverage can be assigned once while running'); END""",
    """CREATE TRIGGER IF NOT EXISTS skill_runs_lifecycle_transition
       BEFORE UPDATE OF status,completed_at,error_type,skip_reason ON skill_runs
       WHEN NOT (
           (NEW.status IS OLD.status AND NEW.completed_at IS OLD.completed_at
            AND NEW.error_type IS OLD.error_type AND NEW.skip_reason IS OLD.skip_reason)
           OR (OLD.status='RUNNING' AND NEW.status='COMPLETED'
               AND NEW.completed_at IS NOT NULL
               AND NEW.error_type IS OLD.error_type AND NEW.skip_reason IS OLD.skip_reason)
           OR (OLD.status='RUNNING' AND NEW.status='FAILED'
               AND NEW.completed_at IS NOT NULL AND NEW.error_type IS NOT NULL
               AND NEW.skip_reason IS OLD.skip_reason)
           OR (OLD.status='RUNNING' AND NEW.status='SKIPPED'
               AND NEW.completed_at IS NOT NULL AND NEW.skip_reason IS NOT NULL
               AND length(trim(NEW.skip_reason))>0 AND NEW.error_type IS OLD.error_type)
       )
       BEGIN SELECT RAISE(ABORT,'invalid SkillRun lifecycle transition'); END""",
    """CREATE TRIGGER IF NOT EXISTS skill_runs_immutable_delete
       BEFORE DELETE ON skill_runs
       WHEN OLD.status IN ('COMPLETED','FAILED','SKIPPED')
       BEGIN SELECT RAISE(ABORT,'SkillRun history is immutable'); END""",
    """CREATE TRIGGER IF NOT EXISTS findings_insert_requires_active_run
       BEFORE INSERT ON findings
       WHEN COALESCE((SELECT status FROM analysis_runs WHERE id=NEW.analysis_run_id),'')
            NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot add a finding to an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS findings_content_update_requires_active_run
       BEFORE UPDATE OF analysis_run_id,skill_run_id,skill_id,skill_version,phenomenon,observation,
           interpretation,alternative_interpretation,claim_types_json,coverage_id,evidence_sufficiency,
           interpretation_confidence,limitations_json,created_at,finalized_at ON findings
       WHEN COALESCE((SELECT status FROM analysis_runs WHERE id=OLD.analysis_run_id),'')
            NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change a finding in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS findings_delete_requires_active_run
       BEFORE DELETE ON findings
       WHEN COALESCE((SELECT status FROM analysis_runs WHERE id=OLD.analysis_run_id),'')
            NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot delete a finding in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_observation_candidates_insert_requires_active_run
       BEFORE INSERT ON finding_observation_candidates
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f
                      ON f.analysis_run_id=ar.id WHERE f.id=NEW.finding_id),'')
            NOT IN ('PENDING','RUNNING')
         OR NEW.status!='UNVERIFIED'
         OR NOT EXISTS(SELECT 1 FROM finding_evidence fe JOIN evidence_artifacts e
                       ON e.id=fe.evidence_id WHERE fe.finding_id=NEW.finding_id
                       AND fe.evidence_id=NEW.evidence_id AND e.kind='TEXTUAL')
       BEGIN SELECT RAISE(ABORT,'observation candidates require active findings and cited textual evidence'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_observation_candidates_update_requires_active_run
       BEFORE UPDATE ON finding_observation_candidates
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f
                      ON f.analysis_run_id=ar.id WHERE f.id=OLD.finding_id),'')
            NOT IN ('PENDING','RUNNING')
         OR COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f
                      ON f.analysis_run_id=ar.id WHERE f.id=NEW.finding_id),'')
            NOT IN ('PENDING','RUNNING')
         OR NEW.status!='UNVERIFIED'
         OR NOT EXISTS(SELECT 1 FROM finding_evidence fe JOIN evidence_artifacts e
                       ON e.id=fe.evidence_id WHERE fe.finding_id=NEW.finding_id
                       AND fe.evidence_id=NEW.evidence_id AND e.kind='TEXTUAL')
       BEGIN SELECT RAISE(ABORT,'observation candidates are unverified and immutable after run completion'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_observation_candidates_delete_requires_active_run
       BEFORE DELETE ON finding_observation_candidates
       WHEN EXISTS(SELECT 1 FROM findings WHERE id=OLD.finding_id)
         AND COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f
                       ON f.analysis_run_id=ar.id WHERE f.id=OLD.finding_id),'')
             NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'observation candidates are immutable after run completion'); END""",
    """CREATE TRIGGER IF NOT EXISTS findings_delete_observation_candidates
       AFTER DELETE ON findings
       BEGIN DELETE FROM finding_observation_candidates WHERE finding_id=OLD.id; END""",
    """CREATE TRIGGER IF NOT EXISTS finding_evidence_insert_requires_active_run
       BEFORE INSERT ON finding_evidence
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=NEW.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change evidence links in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_evidence_update_requires_active_run
       BEFORE UPDATE ON finding_evidence
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=OLD.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change evidence links in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_evidence_delete_requires_active_run
       BEFORE DELETE ON finding_evidence
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=OLD.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change evidence links in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_lineage_insert_requires_active_run
       BEFORE INSERT ON finding_lineage
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=NEW.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change lineage in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_lineage_update_requires_active_run
       BEFORE UPDATE ON finding_lineage
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=OLD.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change lineage in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_lineage_update_requires_active_target_run
       BEFORE UPDATE ON finding_lineage
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=NEW.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change lineage in an immutable analysis run'); END""",
    """CREATE TRIGGER IF NOT EXISTS finding_lineage_delete_requires_active_run
       BEFORE DELETE ON finding_lineage
       WHEN COALESCE((SELECT ar.status FROM analysis_runs ar JOIN findings f ON f.analysis_run_id=ar.id
                      WHERE f.id=OLD.finding_id),'') NOT IN ('PENDING','RUNNING')
       BEGIN SELECT RAISE(ABORT,'cannot change lineage in an immutable analysis run'); END""",
)


def migrate(db: sqlite3.Connection) -> None:
    if db.in_transaction:
        raise MigrationError("cannot migrate while a database transaction is active")

    try:
        current = _current_schema_version(db)
        has_schema_objects = _has_schema_objects(db)
    except sqlite3.Error as exc:
        raise MigrationError(str(exc)) from exc

    if current is None and has_schema_objects:
        raise MigrationError("database contains schema objects but has no applied schema version")

    if current is not None and current > SCHEMA_VERSION:
        raise MigrationError("database schema is newer than this software")

    try:
        db.executescript(f"BEGIN IMMEDIATE;\n{DDL}")
        current = _current_schema_version(db)
        if current is None:
            db.execute("INSERT INTO schema_migrations VALUES(?,datetime('now'))", (SCHEMA_VERSION,))
        elif current < SCHEMA_VERSION:
            _add_column_if_missing(db, "chat_conversations", "research_project_id TEXT")
            db.execute(
                "CREATE INDEX IF NOT EXISTS research_project_documents_project "
                "ON research_project_documents(project_id,document_id)"
            )
            _add_column_if_missing(db, "finding_reviews", "review_order INTEGER")
            _backfill_finding_review_order(db)
            _add_column_if_missing(db, "document_revisions", "ocr_language TEXT")
            _add_column_if_missing(db, "document_revisions", "ocr_dpi INTEGER")
            _add_column_if_missing(
                db, "comparison_runs", "storage_kind TEXT NOT NULL DEFAULT 'json-v1'"
            )
            _add_column_if_missing(db, "reranking_runs", "credential_alias TEXT")
            _add_column_if_missing(db, "reranking_runs", "effective_top_n INTEGER")
            _add_column_if_missing(
                db, "reranking_runs", "returned_count INTEGER NOT NULL DEFAULT 0"
            )
            db.execute(
                "UPDATE reranking_runs SET "
                "candidate_count=(SELECT count(*) FROM retrieval_hits h "
                "WHERE h.retrieval_run_id=reranking_runs.retrieval_run_id), "
                "returned_count=(SELECT count(*) FROM reranking_hits h "
                "WHERE h.reranking_run_id=reranking_runs.id), "
                "effective_top_n=CASE WHEN top_n IS NULL OR top_n>"
                "(SELECT count(*) FROM retrieval_hits h "
                "WHERE h.retrieval_run_id=reranking_runs.retrieval_run_id) "
                "THEN (SELECT count(*) FROM retrieval_hits h "
                "WHERE h.retrieval_run_id=reranking_runs.retrieval_run_id) ELSE top_n END"
            )
            _add_column_if_missing(db, "findings", "analysis_run_id TEXT")
            _add_column_if_missing(db, "findings", "skill_run_id TEXT")
            _add_column_if_missing(db, "findings", "created_at TEXT")
            _add_column_if_missing(db, "findings", "finalized_at TEXT")
            _add_column_if_missing(db, "findings", "coverage_id TEXT")
            db.execute(
                "UPDATE findings SET analysis_run_id=COALESCE(analysis_run_id,'legacy'),"
                "created_at=COALESCE(created_at,datetime('now')) "
                "WHERE analysis_run_id IS NULL OR created_at IS NULL"
            )
            _add_column_if_missing(db, "paragraphs", "page_id TEXT")
            _add_column_if_missing(db, "paragraphs", "text TEXT")
            _add_column_if_missing(db, "paragraphs", "segmentation_version TEXT")
            _add_column_if_missing(db, "paragraphs", "bounding_boxes_json TEXT")
            _add_column_if_missing(db, "document_revisions", "original_path TEXT")
            _add_column_if_missing(db, "document_revisions", "mime_type TEXT")
            _add_column_if_missing(db, "document_revisions", "source_size INTEGER")
            _add_column_if_missing(
                db, "document_revisions", "extraction_status TEXT NOT NULL DEFAULT 'SUCCESS'"
            )
            _migrate_document_revision_identity(db)
            _add_column_if_missing(db, "model_executions", "input_tokens INTEGER")
            _add_column_if_missing(db, "model_executions", "output_tokens INTEGER")
            _add_column_if_missing(db, "model_executions", "reasoning_tokens INTEGER")
            _add_column_if_missing(db, "model_executions", "latency_ms INTEGER")
            _add_column_if_missing(
                db, "model_executions", "parameters_json TEXT NOT NULL DEFAULT '{}'"
            )
            _add_column_if_missing(
                db, "model_executions", "capabilities_json TEXT NOT NULL DEFAULT '{}'"
            )
            _add_column_if_missing(db, "model_executions", "model_price_json TEXT")
            _add_column_if_missing(db, "model_executions", "estimated_cost TEXT")
            _add_column_if_missing(db, "model_executions", "estimated_cost_currency TEXT")
            _add_column_if_missing(db, "model_presets", "reasoning_mode TEXT")
            _add_column_if_missing(db, "model_presets", "reasoning_budget_tokens INTEGER")
            _add_column_if_missing(db, "model_executions", "reasoning_mode TEXT")
            _add_column_if_missing(db, "model_executions", "reasoning_budget_tokens INTEGER")
            _add_column_if_missing(db, "model_executions", "endpoint TEXT")
            _add_column_if_missing(db, "model_executions", "skill_run_id TEXT")
            _add_column_if_missing(db, "skill_runs", "started_at TEXT")
            _add_column_if_missing(db, "skill_runs", "completed_at TEXT")
            _add_column_if_missing(db, "skill_runs", "error_type TEXT")
            _add_column_if_missing(db, "skill_runs", "input_finding_id TEXT")
            _add_column_if_missing(db, "skill_runs", "skip_reason TEXT")
            _add_column_if_missing(db, "artifact_registry", "file_hash TEXT")
            _add_column_if_missing(db, "artifact_registry", "embedding_provider_id TEXT")
            _add_column_if_missing(db, "artifact_registry", "embedding_model TEXT")
            _add_column_if_missing(db, "artifact_registry", "embedding_dimension INTEGER")
            _add_column_if_missing(db, "artifact_registry", "chunker_version TEXT")
            _add_column_if_missing(db, "artifact_registry", "embedding_input_type TEXT")
            _add_column_if_missing(
                db, "source_span_segments", "bbox_status TEXT NOT NULL DEFAULT 'UNAVAILABLE'"
            )
            _add_column_if_missing(
                db,
                "source_span_segments",
                "bbox_projection_version TEXT NOT NULL DEFAULT 'paragraph-block-v1'",
            )
            db.execute(
                "UPDATE source_span_segments SET bbox_status='APPROXIMATE' "
                "WHERE bounding_boxes_json IS NOT NULL AND bounding_boxes_json!='[]'"
            )
            _add_column_if_missing(
                db, "analysis_runs", "software_version TEXT NOT NULL DEFAULT 'unknown'"
            )
            _add_column_if_missing(
                db, "analysis_runs", "project_format_version INTEGER NOT NULL DEFAULT 0"
            )
            _add_column_if_missing(
                db, "analysis_runs", "database_schema_version INTEGER NOT NULL DEFAULT 0"
            )
            db.execute(
                "UPDATE findings SET finalized_at=(SELECT completed_at FROM analysis_runs "
                "WHERE analysis_runs.id=findings.analysis_run_id) "
                "WHERE finalized_at IS NULL AND EXISTS(SELECT 1 FROM analysis_runs "
                "WHERE analysis_runs.id=findings.analysis_run_id AND status='COMPLETED')"
            )
            _make_retrieval_run_context_optional(db)
            db.execute("INSERT INTO schema_migrations VALUES(?,datetime('now'))", (SCHEMA_VERSION,))
        elif current > SCHEMA_VERSION:
            raise MigrationError("database schema is newer than this software")
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS finding_reviews_order "
            "ON finding_reviews(finding_id,review_order)"
        )
        for trigger in IMMUTABILITY_TRIGGERS:
            db.execute(trigger)
        db.commit()
    except sqlite3.Error as exc:
        if db.in_transaction:
            db.rollback()
        raise MigrationError(str(exc)) from exc
    except BaseException:
        if db.in_transaction:
            db.rollback()
        raise


def _current_schema_version(db: sqlite3.Connection) -> int | None:
    has_migration_table = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
    ).fetchone()
    if has_migration_table is None:
        return None
    return db.execute("SELECT max(version) FROM schema_migrations").fetchone()[0]


def _migrate_document_revision_identity(db: sqlite3.Connection) -> None:
    """Replace the old uniqueness key with provenance- and structure-aware identity.

    Legacy text imports can be represented by their canonical hash. Legacy PDFs
    receive that same conservative sentinel; the first import under the explicit
    PDF pipeline manifest will create a new revision instead of claiming that
    unrecorded layout provenance is equivalent.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(document_revisions)")}
    if {"extraction_manifest_json", "structure_hash"} <= columns:
        return

    rows = db.execute(
        "SELECT id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,"
        "text_path,original_path,mime_type,source_size,extraction_status,created_at,"
        "ocr_language,ocr_dpi FROM document_revisions"
    ).fetchall()
    db.execute(
        "CREATE TABLE document_revisions_v49("
        "id TEXT PRIMARY KEY,document_id TEXT NOT NULL,source_hash TEXT NOT NULL,"
        "canonical_hash TEXT NOT NULL,extractor TEXT NOT NULL,coordinate_scheme TEXT NOT NULL,"
        "text_path TEXT NOT NULL,original_path TEXT,mime_type TEXT,source_size INTEGER,"
        "extraction_status TEXT NOT NULL DEFAULT 'SUCCESS',created_at TEXT NOT NULL,"
        "ocr_language TEXT,ocr_dpi INTEGER,"
        "extraction_manifest_json TEXT NOT NULL DEFAULT '{}',"
        "structure_hash TEXT NOT NULL DEFAULT '',"
        "UNIQUE(document_id,source_hash,canonical_hash,extractor,coordinate_scheme,"
        "extraction_manifest_json,structure_hash))"
    )
    for row in rows:
        values = tuple(row)
        extractor = str(values[4])
        coordinate_scheme = str(values[5])
        canonical_hash = str(values[3])
        db.execute(
            "INSERT INTO document_revisions_v49("
            "id,document_id,source_hash,canonical_hash,extractor,coordinate_scheme,"
            "text_path,original_path,mime_type,source_size,extraction_status,created_at,"
            "ocr_language,ocr_dpi,extraction_manifest_json,structure_hash) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (*values, basic_extraction_manifest(extractor, coordinate_scheme), canonical_hash),
        )
    db.execute("DROP TABLE document_revisions")
    db.execute("ALTER TABLE document_revisions_v49 RENAME TO document_revisions")


def _has_schema_objects(db: sqlite3.Connection) -> bool:
    return (
        db.execute(
            "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') "
            "AND name NOT LIKE 'sqlite_%' AND name!='schema_migrations' LIMIT 1"
        ).fetchone()
        is not None
    )


def _add_column_if_missing(db: sqlite3.Connection, table: str, definition: str) -> None:
    column = definition.split()[0]
    columns = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
    if column not in columns:
        db.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")


def _make_retrieval_run_context_optional(db: sqlite3.Connection) -> None:
    """Allow standalone retrievals while preserving analysis-linked retrieval rows."""
    columns = {row[1]: row for row in db.execute("PRAGMA table_info(retrieval_runs)")}
    if not columns:
        raise sqlite3.OperationalError("retrieval_runs table is missing")
    if not columns["analysis_run_id"][3] and not columns["skill_run_id"][3]:
        return

    db.execute(
        "CREATE TABLE retrieval_runs_v43("
        "id TEXT PRIMARY KEY,analysis_run_id TEXT,skill_run_id TEXT,scope_id TEXT NOT NULL,"
        "retrieval_plan_ref TEXT NOT NULL,query_terms_json TEXT NOT NULL,"
        "selection_policy TEXT NOT NULL,candidate_count INTEGER NOT NULL,"
        "inspected_count INTEGER NOT NULL,returned_count INTEGER NOT NULL,"
        "candidate_limit_reached INTEGER NOT NULL,coverage_id TEXT NOT NULL,created_at TEXT NOT NULL,"
        "CHECK((analysis_run_id IS NULL)=(skill_run_id IS NULL)))"
    )
    db.execute(
        "INSERT INTO retrieval_runs_v43 "
        "SELECT id,analysis_run_id,skill_run_id,scope_id,retrieval_plan_ref,query_terms_json,"
        "selection_policy,candidate_count,inspected_count,returned_count,candidate_limit_reached,"
        "coverage_id,created_at FROM retrieval_runs"
    )
    db.execute("DROP TABLE retrieval_runs")
    db.execute("ALTER TABLE retrieval_runs_v43 RENAME TO retrieval_runs")


def _backfill_finding_review_order(db: sqlite3.Connection) -> None:
    """Preserve the pre-migration append order before rowids become disposable."""
    next_order: dict[str, int] = {}
    for row_id, finding_id in db.execute(
        "SELECT rowid,finding_id FROM finding_reviews ORDER BY rowid"
    ).fetchall():
        finding_order = next_order.get(finding_id, 0) + 1
        next_order[finding_id] = finding_order
        db.execute(
            "UPDATE finding_reviews SET review_order=? WHERE rowid=?",
            (finding_order, row_id),
        )
