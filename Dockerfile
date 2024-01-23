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
COPY telegram/ app/telegram/

RUN useradd -m transcriber
RUN chown -R transcriber app/telegram/
RUN mkdir app/models
RUN chown -R transcriber app/models/

USER transcriber

CMD [ "python", "app/telegram/bot.py" ]