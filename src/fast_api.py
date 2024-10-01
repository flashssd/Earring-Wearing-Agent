from src.main import main as jewel_put_on
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, RedirectResponse
from enum import Enum
from pydantic import BaseModel
import traceback

app = FastAPI()


class TypeOption(str, Enum):
    """
    Enum for selecting the type of jewelry to generate.
    """

    type1 = "1"
    type2 = "2"


class JewelryRequest(BaseModel):
    """
    Schema for the jewelry generation request.

    Attributes:
        modelUrl (str): URL of the model image.
        jewelryUrl (str): URL of the jewelry image.
        type (TypeOption): Type of jewelry to generate.
    """

    modelUrl: str
    jewelryUrl: str
    type: TypeOption


async def main(
    model_path_url: str,
    jewel_path_url: str,
    jewelry_type: str,
    test: bool = False,
) -> tuple[str, str]:
    """
    Call the main function for generating the jewelry image and watermark.

    Args:
        model_path_url (str): URL of the model image.
        jewel_path_url (str): URL of the jewelry image.
        jewelry_type (str): Type of jewelry to generate.
        test (bool, optional): Flag to indicate testing mode. Defaults to False.

    Returns:
        Tuple[str, str]: The generated image link and the watermarked image link.
    """
    return jewel_put_on(
        model_path_url,
        jewel_path_url,
        jewelry_type,
        test,
    )


clients: list[WebSocket] = []


@app.get("/")
async def root() -> RedirectResponse:
    """
    Redirects to the jewelry generation page.
    """
    return RedirectResponse(url="/genJewelry")


@app.post("/genJewelry")
async def GenJewelryReq(request: JewelryRequest) -> JSONResponse:
    """
    Handles the jewelry generation request, processes the images, and sends results to WebSocket clients.

    Args:
        request (JewelryRequest): The request payload containing model and jewelry image URLs and the type.

    Returns:
        JSONResponse: The generated image and watermarked image links in JSON format.
    """
    try:
        print(f"Received request: {request.json()}")
        img_link, saltImg_link = await main(
            request.modelUrl,
            request.jewelryUrl,
            request.type.value,
        )
        print(f"Generated image link: {img_link}")
        print(f"Generated watermarked image link: {saltImg_link}")

        for client in clients:
            await client.send_json({"img": img_link, "saltImg": saltImg_link})

        return JSONResponse(content={"img": img_link, "saltImg": saltImg_link})

    except Exception as e:
        error_message = f"Error: {str(e)}"
        traceback_str = traceback.format_exc()
        print(f"{error_message}\n{traceback_str}")

        for client in clients:
            await client.send_json(
                {"status": "error", "message": f"{error_message}\n{traceback_str}"}
            )

        return JSONResponse(
            content={"status": "error", "message": f"{error_message}\n{traceback_str}"},
            status_code=500,
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Manages WebSocket connections and broadcasts messages to all connected clients.

    Args:
        websocket (WebSocket): The WebSocket connection.
    """
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received message: {data}")
    except WebSocketDisconnect:
        clients.remove(websocket)
        print("Client disconnected")
