from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse # collects HTML responses
from fastapi.templating import Jinja2Templates # show template from template response

# css
from fastapi.staticfiles import StaticFiles

import uvicorn

# the app base
app = FastAPI()

# connect templates
templates = Jinja2Templates(directory="templates")

# mount static files on app
app.mount("/static", StaticFiles(directory="static"), name="static")




@app.get('/house/{id}', response_class=HTMLResponse)
async def visit(request: Request, id: int):
    return templates.TemplateResponse("index.html", {"request": request, "id": id})




if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000
