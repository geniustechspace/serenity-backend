import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Define the function to start the FastAPI app
def start_app():
    # Initialize FastAPI app
    app = FastAPI()

    # Allow all origins for CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
        allow_headers=["*"],  # Allow all headers
    )

    # Include the routes from the router
    app.include_router(router)

    return app  # Make sure to return the app instance


# Create the FastAPI app instance
app = start_app()

# Run the application with uvicorn
if __name__ == "__main__":
    logging.info("Starting the application...")  # Use logging correctly
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
        use_colors=True,
    )
