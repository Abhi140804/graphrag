from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    neo4j_uri: str = ""
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    embedding_model: str = ""
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    vector_top_k: int = 8
    graph_hops: int = 3
    fusion_vector_weight: float = 0.55
    fusion_graph_weight: float = 0.45

    @property
    def use_neo4j(self) -> bool:
        return bool(self.neo4j_uri.strip())

    @property
    def use_dense_embeddings(self) -> bool:
        return bool(self.embedding_model.strip())

    @property
    def use_llm(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
