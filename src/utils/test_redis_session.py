import pytest
import asyncio
import os
import uuid
import pytest_asyncio
from utils.redis_session import set_session, get_session, delete_session

pytestmark = pytest.mark.asyncio

@pytest.mark.asyncio
async def test_redis_session_crud():
    session_id = f"test-session-{uuid.uuid4()}"
    data = {"foo": "bar", "num": 42}
    # Set session
    await set_session(session_id, data)
    # Get session
    result = await get_session(session_id)
    assert result == data
    # Delete session
    await delete_session(session_id)
    result2 = await get_session(session_id)
    assert result2 is None 