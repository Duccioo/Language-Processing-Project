FROM python:3.9-slim

# Set global configs
WORKDIR /app
RUN export LC_ALL=C
RUN export LC_CTYPE=C
RUN export LC_NUMERIC=C

# Install system dependencies
RUN apt-get update
RUN apt-get install --no-install-recommends -y \
                    build-essential \
                    ffmpeg \
                    python3-dev \
                    && \
    apt-get clean

# Install Python dependencies
COPY telegram/requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy code and define default command
WORKDIR /app


COPY telegram ./telegram

RUN useradd -m transcriber
RUN chown -R transcriber telegram/
RUN mkdir models
RUN chown -R transcriber models/

USER transcriber

CMD [ "python", "telegram/bot.py" ]