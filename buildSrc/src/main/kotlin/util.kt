import org.gradle.api.Project
import org.gradle.api.file.Directory
import java.io.File
import java.net.URI
import java.net.URL
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.zip.GZIPInputStream

operator fun File.div(other: String) = File(this, other)
operator fun Directory.div(other: String): File = file(other).asFile

val nowFormatted: String
    get() = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"))

infix fun URL.into(file: File) {
    file.outputStream().use { out ->
        openStream().use { `in` -> `in`.copyTo(out) }
    }
}

infix fun URL.gzipInto(file: File) {
    file.outputStream().use { out ->
        GZIPInputStream(openStream()).use { `in` -> `in`.copyTo(out) }
    }
}

var File.text
    get() = readText()
    set(value) = writeText(value)

val URL.text
    get() = readText()

val osName = System.getProperty("os.name")
val os = osName.lowercase()
val arch = System.getProperty("os.arch")
val home = System.getProperty("user.home")

val String.url: URL
    get() = URI(this).toURL()


// provide directly a Directory instead of a DirectoryProperty
val Project.buildDirectory: Directory
    get() = layout.buildDirectory.get()