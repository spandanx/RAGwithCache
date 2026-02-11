import logging
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# def oauth2_scheme():
#     return OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    logging.info("Called verify_password()")
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(username: str, mysqlDB):
    logging.info("Called get_user()")
    result = mysqlDB.get_user_by_username(username)
    if len(result)>0:
        return result[0]

def authenticate_user(username: str, password: str, mysqlDB):
    logging.info("Called authenticate_user()")
    user = get_user(username, mysqlDB)
    logging.info(user)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, SECRET_KEY: str, ALGORITHM: str, expires_delta: str | None = None):
    logging.info("called create_access_token()")
    logging.info(data)
    logging.info(SECRET_KEY)
    logging.info(ALGORITHM)
    logging.info(expires_delta)
    logging.info(type(expires_delta))
    to_encode = data.copy()
    if expires_delta:
        logging.info("expires_delta", expires_delta)
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_delta)
    else:
        logging.info("expires in ", 60, "minutes")
        expire = datetime.now(timezone.utc) + timedelta(minutes=1)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials!",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     token_expired_exception = HTTPException(
#         status_code=status.HTTP_403_FORBIDDEN,
#         detail="Token is expired!",
#         headers={"WWW-Authenticate": "Bearer", "REASON": "TOKEN_EXPIRED"},
#     )
#     db_connection_exception = HTTPException(
#         status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#         detail="Token is expired!",
#         headers={"WWW-Authenticate": "Bearer", "REASON": "DATABASE_ERROR"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         print(payload)
#         username: str = payload.get("sub")
#         print('username', username)
#         if username is None:
#             raise credentials_exception
#         token_data = TokenData(username=username)
#         print(token_data)
#
#     except jwt.ExpiredSignatureError:
#         raise token_expired_exception
#     except InvalidTokenError:
#         raise credentials_exception
#     # try:
#     user = get_user(username=token_data.username)
#     # except Exception as e:
#     #     print(e)
#     print(user)
#     if user is None:
#         raise credentials_exception
#     return user

# async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)],
# ):
#     print(current_user)
#     if current_user['disabled']:
#         raise HTTPException(status_code=400, detail="Inactive user")
#     return current_user

# async def get_all_user():
#     return sorted(list(fake_users_db.keys()))

# async def add_user(user):
#     forbidden_exception_ref_code = HTTPException(
#         status_code=status.HTTP_403_FORBIDDEN,
#         detail="Invalid refferal code!",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     forbidden_exception_user_exists = HTTPException(
#         status_code=status.HTTP_403_FORBIDDEN,
#         detail="User already exists!",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     settings = get_settings()
#     temp_user = dict(user)
#     current_user = get_user(temp_user['username'])
#     if (current_user is not None):
#         raise forbidden_exception_user_exists
#     if 'referral_code' in temp_user:
#         if (temp_user['referral_code'] != settings.REFFERAL_CODE):
#             raise forbidden_exception_ref_code
#     else:
#         raise forbidden_exception_ref_code
#     print("before adding Hashed Pass")
#     print(temp_user)
#     if 'password' in temp_user:
#         temp_user['hashed_password'] = get_password_hash(temp_user['password'])
#         del temp_user['password']
#     print("Update Hashed Pass")
#     print(temp_user)
#     to_be_registered_user = RegisterUser(**temp_user)
#     mysqlDB.create_user(to_be_registered_user)
#     # fake_users_db[temp_user['username']] = temp_user
#     # print(fake_users_db)
#     return temp_user["username"]

if __name__ == "__main__":
    # pass
    passw = "Sample Pass"
    encyrpted = get_password_hash(passw)
    print(encyrpted)
    response = verify_password(passw, encyrpted)
    print(response)
    expires_delta = 2
    changed_datetime = datetime.now(timezone.utc) + timedelta(minutes=expires_delta)
    print(changed_datetime)