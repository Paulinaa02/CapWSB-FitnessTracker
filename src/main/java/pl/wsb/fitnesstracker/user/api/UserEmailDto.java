package pl.wsb.fitnesstracker.user.api;

/**
 * Data Transfer Object for user email search results.
 * Contains only user ID and email address.
 */
public record UserEmailDto(Long id, String email) {

}
