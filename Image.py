from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="Generate an image of a woman wearing a yellow shirt in an office environment with a purple background. The man is looking at his mobile phone, smiling, and cheering with jumping happiness. He has curly hair, white skin tone,. The focus should be on the man and his expressions, with the purple background enhancing the office atmosphere",
  size="1024x1024",
  quality="hd",
  n=1,
)


image_url = response.data[0].url
print(image_url)