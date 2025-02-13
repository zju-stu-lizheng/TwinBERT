# TwinBERT

We propose TwinBERT, a pre-trained language model based framework that leverages anonymized SQL query logs in addition to the conventional database schema information for schema matching.
TwinBERT employ table-level representation to better understand the schema semantics and further pre-train a language model with the historical queries to acquire dataset-specific knowledge.
We perform extensive evaluations on seven open-source datasets with auto-generated SQL logs.



Code structure:

* datasets/ : The dataset used by TwinBERT
* sbert/: SchemaBERT and DistilBERT as baseline
* bert-sql/: SQLBERT pretraining process
* TwinBERT/: The final TwinBERT code
