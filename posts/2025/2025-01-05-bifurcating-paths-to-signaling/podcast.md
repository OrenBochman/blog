1. Convert WAV to MP3

```
ffmpeg -i part1.wav -b:a 192k part1.mp3
```
2. Update the HTML

```
<audio controls="1">
  <source src="part1.mp3" type="audio/mpeg">
</audio>
```


```
<link rel="alternate" type="application/rss+xml" title="Podcast RSS Feed" href="https://your-site.com/feed.xml">
```

```
<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Your Podcast Title</title>
    <link>https://your-site.com</link>
    <description>A brief description of your podcast.</description>
    <language>en-us</language>
    <itunes:author>Your Name</itunes:author>
    <itunes:explicit>no</itunes:explicit>
    <itunes:image href="https://your-site.com/logo.jpg" />
    <item>
      <title>Episode 1: Title</title>
      <description>A brief description of the episode.</description>
      <link>https://your-site.com/episodes/episode1.mp3</link>
      <enclosure url="https://your-site.com/episodes/episode1.mp3" type="audio/mpeg" length="1234567" />
      <guid>https://your-site.com/episodes/episode1.mp3</guid>
      <pubDate>Thu, 28 Nov 2024 12:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
```